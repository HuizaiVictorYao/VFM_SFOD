# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch_sfda
from models import build_model
from models.dinoV2 import load_dinov2
from models.GroundingDINO.groundingdino.util.inference import load_model
import copy
import os

GDINO_CONFIG_PATH = "./models/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_WEIGHTS_PATH = "./models/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_classifier_names', default=['class_embed'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_sfda', dest='dataset_sfda',
                      help='SFDA dataset setting',
                      default='city_to_foggy', type=str)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/cityscape', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    parser.add_argument(
        "--pretrained_backbone_path",
        default="",
        type=str,
    )
    parser.add_argument("--drop_path_rate", default=0.2, type=float)

    # SFDA options
    parser.add_argument('--VFM_SFOD', default=False, action='store_true', help='whether to apply our VFM-enhanced SFOD')
    parser.add_argument('--conf_thres', default=0.3, type=float, help='confidence threshold for psl')
    parser.add_argument('--cont_inst', default=False, action='store_true', help='whether to use instance-contrastive loss')
    parser.add_argument('--cont_inst_loss_coef', default=1.0, type=float)
    parser.add_argument('--ddis', default=False, action='store_true', help='whether to use dino distill loss')
    parser.add_argument('--ddis_loss_coef', default=0.5, type=float)
    parser.add_argument('--gdino', default=False, action='store_true', help='whether to use grounding dino')
    parser.add_argument('--test_target', default=False, action='store_true', help='test adapted model parameters')
    # matcher type
    parser.add_argument('--matcher', default='hungarian', type=str)

    return parser

def set_dataset_args(args):
    # Set sfda datasets
    # More datasets to be added
    bb_name = ""
    if "swin" in args.backbone:
        bb_name = "_swin"+args.backbone[5]
    elif "vit" in args.backbone:
        bb_name = "_vit"+args.backbone[4]
    elif "resnet" in args.backbone:
        bb_name = "_"+args.backbone
    if args.dataset_sfda == "synthetic_to_real":
        args.dataset_src = 'sim'
        args.dataset_tgt = 'cityscape_car'
        args.coco_path = './data/' + args.dataset_tgt
        args.output_dir = 'exps/sim2real'
        args.num_classes = 2

    if args.dataset_sfda == "city_to_foggy":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'foggy'
        args.coco_path = './data/' + args.dataset_tgt
        args.output_dir = 'exps/city2foggy'
        args.num_classes = 9

    if args.dataset_sfda == "city_to_bdd":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'bdd_coco'
        args.coco_path = './data/' + args.dataset_tgt
        args.output_dir = 'exps/city2bdd'
        args.num_classes = 8

    if args.dataset_sfda == "city_to_rainy":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'rainy'
        args.coco_path = './data/' + args.dataset_tgt
        args.output_dir = 'exps/city2rainy'
        args.num_classes = 9

    if args.dataset_sfda == "kitti_to_city":
        args.dataset_src = 'kitti'
        args.dataset_tgt = 'cityscape_car'
        args.coco_path = './data/' + args.dataset_tgt
        args.output_dir = 'exps/kitti2city'
        args.num_classes = 2
    
    if args.dataset_sfda == "city_to_acdc_fog":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'acdc_fog'
        args.coco_path = './data/acdc/fog'
        args.output_dir = 'exps/city2acdc_fog'
        args.num_classes = 9

    if args.dataset_sfda == "city_to_acdc_night":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'acdc_night'
        args.coco_path = './data/acdc/night'
        args.output_dir = 'exps/city2acdc_night'
        args.num_classes = 9

    if args.dataset_sfda == "city_to_acdc_rain":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'acdc_rain'
        args.coco_path = './data/acdc/rain'
        args.output_dir = 'exps/city2acdc_rain'
        args.num_classes = 9

    if args.dataset_sfda == "city_to_acdc_snow":
        args.dataset_src = 'cityscapes'
        args.dataset_tgt = 'acdc_snow'
        args.coco_path = './data/acdc/snow'
        args.output_dir = 'exps/city2acdc_snow'
        args.num_classes = 9

    args.output_dir = args.output_dir+bb_name
    args.resume = args.output_dir + '/source.pth'
    args.pretrained_backbone_path = args.resume
    if args.test_target:
        args.resume = args.output_dir + '/target.pth'

def set_sfod_args(args):
    if args.VFM_SFOD:
        args.cont_inst = True
        args.ddis = True
        args.gdino = True
    else:
        args.cont_inst = False
        args.ddis = False
        args.gdino = False

def load_gdino_with_prompts(dataset):
    from models.GroundingDINO.groundingdino.util.vl_utils import build_captions_and_token_span
    gdino = load_model(GDINO_CONFIG_PATH, GDINO_WEIGHTS_PATH)
    category_names = [v['name'] for k, v in sorted(dataset.coco.cats.items())]
    text_prompt, token_span = build_captions_and_token_span(category_names, force_lowercase=True)
    return gdino, text_prompt, token_span



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val,
                                                          shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val,
                                                      shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": # keep the classifier frozen following previous DAOD approaches
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and not match_name_keywords(n, args.lr_classifier_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts,
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected_keys = [
            k for k in unexpected_keys
            if not (k.endswith('total_params') or k.endswith('total_ops'))
        ]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval,
                                 output_dir / "eval.pth")
        return

    model_teacher = copy.deepcopy(model)

    dinov2 = None
    prototypes = None
    velocity = None
    if args.ddis:
        dinov2_size = "g"
        dinov2 = load_dinov2(dinov2_size, freeze_params=True)
        dinov2.to(device)
        dinov2.eval()
        prototypes = [None] * args.num_classes
        velocity = [None] * args.num_classes

    gdino = None
    text_prompt = None
    token_span = None
    if args.gdino:
        gdino, text_prompt, token_span = load_gdino_with_prompts(dataset_val)

    print("Start training")
    start_time = time.time()

    itl = [args.conf_thres for _ in range(args.num_classes)]
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats,prototypes,velocity = train_one_epoch_sfda(
            model,
            model_teacher,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            initial_thres_list=itl,
            dinov2=dinov2,
            gdino=gdino,
            text_prompt=text_prompt,
            token_span=token_span,
            args=args,
            prototypes=prototypes,
            velocity=velocity)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir /
                                        f'target_checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
            # save teacher model
            checkpoint_paths = [output_dir / 'checkpoint_t.pth']
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir /
                                        f'target_checkpoint{epoch:04}_t.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            'model': model_teacher.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)

        test_stats_t, coco_evaluator_t = evaluate(model_teacher, criterion,
                                                  postprocessors,
                                                  data_loader_val, base_ds,
                                                  device, args.output_dir)

        log_stats = {
            **{
                f'train_{k}': v
                for k, v in train_stats.items()
            },
            **{
                f'test_{k}': v
                for k, v in test_stats.items()
            }, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    def fix_all_seeds(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True

    fix_all_seeds(42)
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_sfod_args(args)
    set_dataset_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
