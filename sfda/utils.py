import torch
from collections import OrderedDict

import torch.nn.functional as F
from torchvision.ops import roi_align, nms
import numpy as np

from torchvision import transforms
from torchvision import transforms as T

from PIL import Image
import cv2

from typing import Iterable
import time

from sfda.fusion import entropy_weighted_fusion

def get_prototype(model_teacher: torch.nn.Module, data_loader: Iterable, device: torch.device, dinov2, conf_thres: float = 0.8, num_classes: int = 91):
    model_teacher.eval()
    prototype_list = [(None, 0) for _ in range(num_classes - 1)]  # Foreground prototypes and counts.

    start_time = time.time()
    for ii, batch in enumerate(data_loader):
        if ii % 10 == 0:
            elapsed_time = time.time() - start_time
            batches_left = len(data_loader) - ii
            eta = (elapsed_time / (ii + 1)) * batches_left if ii > 0 else 0
            print(f"Processing batch {ii}/{len(data_loader)}... ETA: {eta:.2f} seconds")
        samples, targets = batch
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        outputs, base_feat_teacher, _, enc_feats, hs = model_teacher(samples, return_base_feat=True)

        all_scale_logits = [aux_output['pred_logits'] for aux_output in outputs['aux_outputs']]
        all_scale_logits.append(outputs['pred_logits'])
        all_scale_boxes = [aux_output['pred_boxes'] for aux_output in outputs['aux_outputs']]
        all_scale_boxes.append(outputs['pred_boxes'])

        logits = torch.cat(all_scale_logits, dim=1)  # (bs, N, num_classes)
        boxes = torch.cat(all_scale_boxes, dim=1)    # (bs, N, 4)

        probs = torch.sigmoid(logits)

        for i in range(probs.shape[0]):  # batch
            prob = probs[i]  # (N, C)
            box = boxes[i]   # (N, 4)
            image = samples.tensors[i].unsqueeze(0)  # (1, 3, H, W)

            is_fg = get_binary_predictions(prob)
            max_probs, max_classes = prob[:, 1:].max(dim=1)
            max_classes = max_classes + 1  # adjust index: class 1..C

            foreground_mask = (is_fg & (max_probs > conf_thres))
            fg_indices = torch.nonzero(foreground_mask).squeeze(1)

            if fg_indices.numel() == 0:
                continue

            selected_boxes = box[fg_indices]  # (M, 4)
            selected_classes = max_classes[fg_indices]  # (M,)

            for j in range(selected_boxes.size(0)):
                class_id = selected_classes[j].item()
                if class_id == 0 or class_id > num_classes - 1:
                    continue  # skip background or invalid

                box_xyxy = xywh_to_xyxy(selected_boxes[j].unsqueeze(0))  # (1, 4), normalized
                box_abs = box_xyxy[0].clone()
                box_abs[0::2] *= image.shape[-1]  # width
                box_abs[1::2] *= image.shape[-2]  # height
                box_abs = box_abs.clamp(0, max(image.shape[-1], image.shape[-2]))  # clamp

                x1, y1, x2, y2 = box_abs.int().tolist()
                if x2 <= x1 or y2 <= y1:
                    continue  # skip invalid box

                crop = image[:, :, y1:y2, x1:x2]  # (1, 3, h, w)
                crop_resized = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)

                with torch.no_grad():
                    feat = dinov2(crop_resized)['x_norm_clstoken']  # assumed to output (1, D)
                    feat = feat.squeeze(0)

                proto, count = prototype_list[class_id - 1]
                if proto is None:
                    prototype_list[class_id - 1] = (feat.clone(), 1)
                else:
                    new_count = count + 1
                    updated_proto = (proto * count + feat) / new_count
                    prototype_list[class_id - 1] = (updated_proto, new_count)

    model_teacher.train()

    # Return class prototypes (prototype only).
    final_prototypes = [proto for proto, count in prototype_list]
    for idx, proto in enumerate(final_prototypes):
        if proto is not None:
            print(f"Class {idx + 1}: Prototype shape: {proto.shape}")
        else:
            print(f"Class {idx + 1}: Prototype is None")
    return final_prototypes

def prepare_psl(outputs, targets, threshold: list = None, get_all=False, gdino_preds=None):
    """
    fuction: convert (teacher) outputs to targets (pseudo labels)
    outputs: Teacher DETR outputs
        Is a tuple with
         'pred_logits' : [b,m,k]
         'pred_boxes' : [b,m,4] tensor
         among which b is batchsize, m is bbox number, k is class number
    targets: formatted optimization target for detr
        Is a list of tuples, with list length = b
        In which a tuple is a target for the current image in minibatch
        with
         'boxes', [mb,4], mb is gt bbox number
         'labels', [mb], labels int
         'image_id', [1] tensor, image_id from coco anns
         'area', bbox area, [mb] tensor and each is calc by
                            h*w*size[0]*size[1](not orig_size)
         'iscrowd', if is crowd, here we only need [mb] zeros
         'orig_size', 'size' is the size for images, here we directly
                            obtain them from targets['orig_size'], t-
                            argets['size'] since this is not supervi-
                            sory signals

    p.s. bbox format in outputs and targets are both normalized [x,y,w,h]

    p.s. we only extract metadatas IRRELEVANT to labels in original gt targets, 
         which would NOT cause information leakage to pseudo labels
         such as image_id, orig_size(original size of the image), etc.
    """
    batchsize = outputs['pred_logits'].shape[0]
    targets_psl = [{} for _ in range(batchsize)]

    image_id_list = [target['image_id'] for target in targets]
    orig_size_list = [target['orig_size'] for target in targets]
    size_list = [target['size'] for target in targets]

    boxes_list, labels_list, probs_list = process_predictions(outputs['pred_logits'], outputs['pred_boxes'], threshold, get_all, do_nms=True, gdino_preds=gdino_preds)

    area_list = calc_area(boxes_list, size_list)
    iscrowd_list = [target['iscrowd'] for target in targets]

    for cur in range(batchsize):
        targets_psl_cur = targets_psl[cur]
        targets_psl_cur['boxes'] = boxes_list[cur]
        targets_psl_cur['labels'] = labels_list[cur]
        targets_psl_cur['image_id'] = image_id_list[cur]
        targets_psl_cur['area'] = area_list[cur]
        targets_psl_cur['iscrowd'] = iscrowd_list[cur]
        targets_psl_cur['orig_size'] = orig_size_list[cur]
        targets_psl_cur['size'] = size_list[cur]
        if not get_all:
            targets_psl_cur['probs'] = probs_list[cur]

    return targets_psl


def get_binary_predictions(predictions):
    # get the idx with highest prediction scores in each tensor
    max_indices = torch.argmax(predictions, dim=1)
    binary_predictions = torch.ones_like(max_indices, dtype=torch.bool)
    binary_predictions[max_indices == 0] = False
    return binary_predictions

def update_prototypes_with_momentum(
    patch_feats, H, W,
    boxes_list,
    prototypes,
    velocities,
    output_size=7,
    mu=0.9,
    eta=0.1
):
    """
    patch_feats: [B, H*W, C] patch features
    H, W: feature map size
    boxes_list: list of dicts, len=B, each with:
      'boxes': [N, 4] normalized xywh
      'labels': [N] class ids
    prototypes: list, len=num_classes, each is [C] or None
    velocities: list, len=num_classes, each is [C] or None
    output_size: ROIAlign output size (default 7)
    mu, eta: momentum parameters
    """

    device = patch_feats.device
    b, HW, c = patch_feats.shape
    num_classes = len(prototypes)

    # Reshape to [B, C, H, W].
    patch_feats = patch_feats.permute(0, 2, 1).contiguous()  # [b, c, H*W]
    patch_feats = patch_feats.view(b, c, H, W)               # [b, c, H, W]

    # Build ROIAlign boxes in feature coordinates.
    rois = []
    roi_labels = []

    for img_idx, info in enumerate(boxes_list):
        boxes = info['boxes']  # [n,4], xywh norm
        labels = info['labels']  # [n]

        if boxes.numel() == 0:
            continue

        # Convert normalized xywh to xyxy in feature coords.
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        x2 = (cx + 0.5 * w) * W
        y2 = (cy + 0.5 * h) * H

        cur_rois = torch.stack([x1, y1, x2, y2], dim=-1)
        batch_inds = torch.full((cur_rois.shape[0], 1), img_idx, dtype=cur_rois.dtype, device=cur_rois.device)
        cur_rois = torch.cat([batch_inds, cur_rois], dim=-1)  # [n,5]
        rois.append(cur_rois)
        roi_labels.append(labels)

    if len(rois) == 0:
        return prototypes, velocities

    rois = torch.cat(rois, dim=0).to(device)        # [total_rois,5]
    roi_labels = torch.cat(roi_labels, dim=0).to(device)  # [total_rois]

    # ROIAlign on patch features.
    aligned_feats = roi_align(
        patch_feats, rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True
    )  # [total_rois, c, output_size, output_size]

    # Average ROI features.
    aligned_feats = aligned_feats.mean(dim=[2, 3])  # [total_rois, c]
    # Update per-class prototypes.
    for cls_id in range(num_classes):
        cls_mask = (roi_labels == cls_id)
        if not torch.any(cls_mask):
            continue

        cls_feats = aligned_feats[cls_mask]  # [n_cls, c]
        f_t = cls_feats.mean(dim=0)  # [c]

        if prototypes[cls_id] is None:
            prototypes[cls_id] = f_t.detach()
            velocities[cls_id] = torch.zeros_like(f_t)
        else:
            velocity = velocities[cls_id]
            prototype = prototypes[cls_id]

            velocity = mu * velocity + (1 - mu) * (f_t - prototype)
            prototype = prototype + eta * velocity

            prototypes[cls_id] = prototype.detach()
            velocities[cls_id] = velocity.detach()

    return prototypes, velocities

def process_predictions(logits, boxes, threshold: list = None, get_all=False, do_nms=False, gdino_preds=None):
    """
    Func: Process preds with confidence higher than thres to pseudo labels

    (b: batchsize;m: bounding box num; k: class num)
    Input:  
        logits, detr's outputs['pred_logits'], [b,m,k]
        boxes, outputs['pred_boxes'], [b,m,4]
        threshold, hyperparam
    Output:
        selected_boxes, current chosen boxes
            list with length=b, [mi,4]; 0<=mi<=m
        selected_classes, current chosen classes idx
            list with length=b, [mi]; 0<=mi<=m
        selected_probs, current chosen class probs
            list with length=b, [mi,k]; 0<=mi<=m 
    """
    if get_all:
        selected_boxes = [boxes[i] for i in range(boxes.shape[0])]
        cls_tensor = torch.argmax(logits.sigmoid(), dim=2)
        selected_classes = [cls_tensor[i] for i in range(cls_tensor.shape[0])]
        return selected_boxes, selected_classes, None, None
    # Step 1: get probs
    threshold = [0.3 for _ in range(logits.shape[2])]
    threshold_low = [0.2 for _ in range(logits.shape[2])]
    probabilities = torch.sigmoid(logits)

    selected_indices = []
    for prob in probabilities:
        is_fg = get_binary_predictions(prob)

        max_probs, max_indices = prob[:, 1:].max(dim=1)
        thres_for_max_probs = torch.zeros([max_indices.shape[0]]).cuda()
        thres_for_unc_probs = torch.zeros([max_indices.shape[0]]).cuda()
        for idx in range(max_indices.shape[0]):
            thres_for_max_probs[idx] = threshold[max_indices[idx]+1]# prob[:, 1:], idx+=1
            thres_for_unc_probs[idx] = threshold_low[max_indices[idx]+1]
        fg_prob_higher_than_thres = max_probs > thres_for_max_probs

        foreground_indices = torch.nonzero(is_fg & fg_prob_higher_than_thres)[:, 0]
        selected_indices.append(foreground_indices)

    # Step 3: Extract bbox predictions from selected indices.
    selected_boxes = []
    for i in range(len(selected_indices)):
        selected_boxes.append(boxes[i][selected_indices[i]])

    selected_probs = []
    for i in range(len(selected_indices)):
        selected_probs.append(probabilities[i][selected_indices[i]])

    # Step 4: obtain index of the highest class prob of each bounding box
    max_class_indices = probabilities[:, :, 1:].argmax(dim=2)

    # Step 5: obtain corresponding classes in max_class_indices
    selected_classes = [max_class_indices[i, j]+1 for i, j in enumerate(selected_indices)]


    if gdino_preds is not None:
        gdino_boxes, gdino_logits, gdino_phrases, gdino_probs, gdino_text_prompt = gdino_preds   
        category_names = [name.strip() for name in gdino_text_prompt.split('.') if name.strip()]
        # Build category name to index map.
        category_to_index = {name: idx for idx, name in enumerate(category_names)}

        for i_batch in range(len(selected_boxes)):
            cur_boxes = selected_boxes[i_batch]
            cur_classes = selected_classes[i_batch]
            cur_probs = selected_probs[i_batch]
            cur_gdino_boxes = gdino_boxes[i_batch]

            gdino_labels = [category_to_index[phrase] for phrase in gdino_phrases[i_batch]]
            gdino_labels = [x+1 for x in gdino_labels]
            gdino_labels = torch.tensor(gdino_labels).to(cur_boxes.device).to(torch.int)# +1 for background
            cur_gdino_classes = gdino_labels

            gdino_probs_fg = gdino_probs[i_batch]
            gdino_probs_fg = F.softmax(gdino_probs_fg, dim=1)
            zeros_column = torch.zeros(
                gdino_probs_fg.size(0),
                1,
                device=gdino_probs_fg.device,
                dtype=gdino_probs_fg.dtype,
            )  # foreground boxes, background prob is 0
            gdino_probs_fg_bg = torch.cat([zeros_column, gdino_probs_fg], dim=1)
            cur_gdino_probs = gdino_probs_fg_bg

            selected_boxes[i_batch], selected_probs[i_batch], selected_classes[i_batch] = entropy_weighted_fusion(cur_boxes, cur_probs, cur_classes, 
                                 cur_gdino_boxes, cur_gdino_probs, cur_gdino_classes, 
                                 iou_thr=0.7)

    # Step 6: NMS
    if do_nms:
        for i_batch in range(len(selected_boxes)):
            cur_boxes = selected_boxes[i_batch]
            cur_classes = selected_classes[i_batch]
            cur_probs = selected_probs[i_batch]

            scores = cur_probs[:, 1:].max(dim=1).values

            keep_indices = nms(xywh_to_xyxy(cur_boxes), scores, iou_threshold=0.6)

            selected_boxes[i_batch] = cur_boxes[keep_indices]
            selected_classes[i_batch] = cur_classes[keep_indices]
            selected_probs[i_batch] = cur_probs[keep_indices]
    return selected_boxes, selected_classes, selected_probs




def calc_area(boxes_list,sizes_list):
    """
    just calc area value in targets union, actually no use
    """
    norm_areas_list = [[box[2]*box[3] for box in boxes] for boxes in boxes_list]

    area_list = [[norm_area*sizes[0]*sizes[1] for norm_area in norm_areas] for norm_areas, sizes in zip(norm_areas_list,sizes_list)]

    area_tensor_list = [torch.tensor(areas).cuda() for areas in area_list]

    return area_tensor_list

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.999, shared_name_list = []):

    student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if len(shared_name_list) > 0:
            for name in shared_name_list:
                if name in key:
                    continue 
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict


def format_boxes(pseudo_boxes_list, h, w):
    """
    xywh to xyxy
    """
    for b in range(len(pseudo_boxes_list)):
        if pseudo_boxes_list[b].shape[0] != 0:
            pseudo_boxes_list[b] = xywh_to_xyxy(pseudo_boxes_list[b])
            pseudo_boxes_list[b][:, [0, 2]] *= w  
            pseudo_boxes_list[b][:, [1, 3]] *= h 

    return pseudo_boxes_list

def xywh_to_xyxy(boxes):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    converted_boxes = torch.stack((x1, y1, x2, y2), dim=1)

    return converted_boxes

def xyxy_to_xywh(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2

    converted_boxes = torch.stack((x, y, w, h), dim=1)

    return converted_boxes

@torch.no_grad()
def visualization(samples, results=None, save_path='./TODO.jpg', gt=False, gt_targets=None):

    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    toPIL = transforms.ToPILImage()
    scale_factor = 2  # or 3, depending on needs

    for b in range(samples.tensors.shape[0]):
        image_tensor = inv_normalize(samples.tensors[b].clone())
        img_pil = toPIL(image_tensor)
        orig_size = img_pil.size
        new_size = (orig_size[0] * scale_factor, orig_size[1] * scale_factor)
        img_pil = img_pil.resize(new_size, Image.BICUBIC)

        if results is not None:
            boxes_all = results[b]['boxes']
            scores_all = results[b]['scores']
            labels_all = results[b]['labels']
            indices = torch.nonzero(scores_all > 0.3).squeeze()

            if gt:
                boxes_all = gt_targets[b]['boxes']
                labels_all = gt_targets[b]['labels']
                indices = torch.nonzero(labels_all > -999).squeeze()
                scores_all = torch.ones_like(labels_all, dtype=torch.float32)

            if indices.numel() >= 1:
                try:
                    boxes_valid = boxes_all[indices] * scale_factor
                    scores_valid = scores_all[indices]
                    labels_valid = labels_all[indices]
                    img_pil = draw_boxes_with_confidence(img_pil, boxes_valid, scores_valid, labels_valid, gt=gt)
                except:
                    continue

        img_pil.save(save_path.replace(".jpg", ".png"))


def draw_boxes_with_confidence(
    pil_image,
    boxes,
    confidences,
    labels,
    gt=False,
    class_names=None,
    font_scale=1.0,
    font_thickness=2,
    rect_thickness=4
):
    """
    Draw bounding boxes with label and confidence score using OpenCV.

    Args:
        pil_image: PIL.Image
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
        confidences: Tensor [N], optional if gt=True
        labels: Tensor [N]
        gt: If True, no confidence shown
        class_names: Optional list of class names
        font_scale: Font size for cv2.putText (default 1.0)
        font_thickness: Font stroke thickness (default 2)
        rect_thickness: Bounding box thickness (default 3)
    """
    image = np.array(pil_image)

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = boxes.cpu().numpy().astype(int)
    if not gt:
        confidences = confidences.cpu().numpy()
    labels = labels.cpu().numpy()

    if class_names is None:
        class_names = [
            "None", "person", "car", "train", "rider",
            "truck", "motor", "bicycle", "bus", "unknown"
        ]

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0)
    ]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = labels[i]
        color = colors[label % len(colors)]

        if not gt:
            confidence = confidences[i]
            class_name = class_names[label] if label < len(class_names) else f"cls_{label}"
            text = f"{class_name}: {confidence:.2f}"
        else:
            class_name = class_names[label] if label < len(class_names) else f"cls_{label}"
            text = class_name

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, rect_thickness)

        # Draw text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)

        # Draw text
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
