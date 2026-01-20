# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.nn.functional as F
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

from sfda.utils import prepare_psl, update_teacher_model, update_prototypes_with_momentum
from sfda.augmentation import strong_aug

import copy

import numpy as np
import itertools
from terminaltables import AsciiTable

import torchvision.transforms as T
from models.GroundingDINO.groundingdino.util.inference import predict_batch

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_sfda(model: torch.nn.Module,
                         model_teacher: torch.nn.Module,
                         criterion: torch.nn.Module,
                         data_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epoch: int,
                         max_norm: float = 0,
                         initial_thres_list: list = None,
                         args=None,
                         dinov2=None,
                         gdino=None,
                         text_prompt=None,
                         token_span=None,
                         prototypes=None,
                         velocity=None):

    model.train()
    model_teacher.train()
    for param in model_teacher.parameters():
        param.requires_grad = False
    thres_list = [0.3] * len(initial_thres_list)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter(
        'grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    step = 0

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq,
                                     header):
        wa_samples = copy.deepcopy(samples)
        sa_samples = copy.deepcopy(samples)

        sa_samples.tensors = strong_aug(wa_samples.tensors.detach().clone())

        # Use weakly augmented images to compute DINO features for distillation.
        dino_feat = None
        if dinov2 is not None:
            dinov2_size = "g"

            def get_dino_layer_indices(dinov2_size: str):
                """Return intermediate layer indices for a given ViT size."""
                size2depth = {"s": 12, "b": 12, "l": 24, "g": 40}
                depth = size2depth[dinov2_size]
                return [int(depth * i / 5) for i in range(1, 4)]  # e.g. [8, 16, 24] for g
            transform = T.Compose([
                T.Resize((224, 224),
                         interpolation=T.InterpolationMode.BICUBIC),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])
            dino_input = transform(wa_samples.tensors)

            layer_indices = get_dino_layer_indices(dinov2_size)
            with torch.no_grad():
                intermediate_feats = dinov2.get_intermediate_layers(
                    dino_input,
                    n=layer_indices,
                    reshape=False,
                    return_class_token=False,  # patch tokens only
                )  # List of 3 tensors: [B, N_patches, C]

                final_out = dinov2(dino_input)
                final_feat = final_out["x_norm_patchtokens"]  # [B, N_patches, C]

                dino_feats = list(intermediate_feats) + [final_feat]  # list of 4 tensors
                final_out["x_norm_patchtokens"] = torch.stack(dino_feats, dim=0).mean(dim=0)
                dino_feat = final_out

        outputs, base_feat, _, enc_feats_s, _ = model(
            sa_samples, return_base_feat=True)

        last_base_feat_aligned = None
        if dinov2 is not None:
            last_base_feat = base_feat[-1].tensors
            last_base_feat_aligned = F.adaptive_avg_pool2d(
                last_base_feat, (16, 16))
            last_base_feat_aligned = last_base_feat_aligned.flatten(2).transpose(1, 2)
            last_base_feat_aligned = model.module.dino_distill_mlp(
                last_base_feat_aligned)
        gdino_preds = None
        if gdino is not None:
            box_threshold = 0.35
            text_threshold = 0.25
            boxes, logits, phrases, probs, _ = predict_batch(
                model=gdino,
                images=wa_samples,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                remove_combined=True,
                token_span=token_span,
            )
            gdino_preds = [boxes, logits, phrases, probs, text_prompt]
        with torch.no_grad():
            outputs_teacher, _, _, _, _ = model_teacher(
                wa_samples, return_base_feat=True)
        with torch.no_grad():
            targets = prepare_psl(
                outputs_teacher,
                targets,
                threshold=thres_list,
                gdino_preds=gdino_preds)
        if dinov2 is not None:
            with torch.no_grad():
                prototypes, velocity = update_prototypes_with_momentum(
                    dino_feat['x_norm_patchtokens'],
                    H=16,
                    W=16,
                    boxes_list=targets,
                    prototypes=prototypes,
                    velocities=velocity)

        roi_feat_l = []
        if args.ddis:
            b, h, w, c = enc_feats_s[-1].shape
            x_flat = enc_feats_s[-1].reshape(-1, c)
            x_proj = model.module.dino_distill_mlp_instance(x_flat)
            x_proj = x_proj.view(b, h, w, -1)  # [b, h, w, dim_dinov2]
            for bb in range(x_proj.shape[0]):
                roi_feat_l.append(x_proj[bb])

        roi_feat_ts_list = [roi_feat_l, prototypes]

        loss_dict = criterion(
            outputs,
            targets,
            roi_feat_ts_list=roi_feat_ts_list,
            dino_distill_feat=[dino_feat, last_base_feat_aligned])

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Ensure DDP sees distillation MLP parameters as used.
        param_sum = torch.tensor(0., device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if "dino_distill_mlp" in name:
                param_sum += (param * 0).sum()

        losses += param_sum

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if step % 5 == 0:
            with torch.no_grad():
                new_teacher_dict = update_teacher_model(
                    model,
                    model_teacher,
                    keep_rate=0.999,
                    shared_name_list=["roi_layer", "query_mlp", "MHA"])
                model_teacher.load_state_dict(new_teacher_dict)
                for k, v in model_teacher.named_parameters():
                    v.requires_grad = False
        step += 1

        samples, targets = prefetcher.next()
    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }, prototypes, velocity

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, use_tsne=False):
    tsne_feats = []
    tsne_labels = []
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0.5]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, _, _, _, hs = model(samples, return_base_feat=True)

        if use_tsne:
            feat = hs[-1]
            prob = outputs["pred_logits"].softmax(-1)
            scores, labels = prob.max(-1)
            # Keep confident, non "no-object" queries for t-SNE.
            for i in range(feat.shape[0]):
                for j in range(feat.shape[1]):
                    if labels[i, j] != prob.shape[-1] - 1 and scores[i, j] > 0.3:
                        tsne_feats.append(feat[i, j].cpu().numpy())
                        tsne_labels.append(labels[i, j].cpu().item())

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        classwise=True
        if classwise:
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa

            cocoEval = coco_evaluator.coco_eval['bbox']
            coco = coco_evaluator.coco_eval['bbox'].cocoDt

            precisions = cocoEval.eval['precision']
            catIds = coco.getCatIds()
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(catIds) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(catIds):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float('nan')
                results_per_category.append(
                    ('{}'.format(nm['name']),
                     '{:0.3f}'.format(float(ap * 100))))

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (N_COLS // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print(table.table)

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    if use_tsne:
        if len(tsne_feats) > 0:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns

            print("Running t-SNE on collected features...")
            tsne = TSNE(n_components=2, perplexity=50, init='pca', learning_rate='auto')
            tsne_result = tsne.fit_transform(np.array(tsne_feats))

            plt.figure(figsize=(8, 8))
            palette = sns.color_palette("hls", len(set(tsne_labels)))
            sns.scatterplot(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                hue=tsne_labels,
                palette=palette,
                legend=False,
                s=10
            )
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')
            plt.tight_layout()
            plt.savefig("./tsne_tgt.png", bbox_inches='tight', pad_inches=0.1)
            plt.close()
    return stats, coco_evaluator
    