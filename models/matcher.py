# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            # Also concat the target labels and boxes

            # print(targets)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            #print(C.shape) torch.Size([600, 5])
            C = C.view(bs, num_queries, -1).cpu()
            #print(C.shape) torch.Size([2, 300, 5])
            sizes = [len(v["boxes"]) for v in targets]
            #print(sizes) [batchsize]

            # for i, c in enumerate(C.split(sizes, -1)):
            #     print(c[i].shape)
            # print("=======================")
            # torch.Size([300, 19])
            # torch.Size([300, 10])

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            # for idx in range(len(indices)):
            #     indices[idx] = tuple(np.concatenate((t, t)) for t in indices[idx])

            #print([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])
            # [(tensor([], dtype=torch.int64), tensor([], dtype=torch.int64)), (tensor([ 11,  62, 251, 264, 286]), tensor([2, 4, 1, 3, 0]))]
            # [(tensor([  6,  98, 147, 150, 240]), tensor([4, 3, 2, 0, 1])), (tensor([ 82, 158, 194, 280, 294]), tensor([3, 0, 4, 1, 2]))]
            # print(indices)

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

"""NOTE: This class is NOT used in FRANCk"""
class O2MMatcher(nn.Module):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 candidate_topk: int = 13):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.candidate_topk = candidate_topk
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, alpha=1, beta=6):
        """
        o2m matching 

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                               ** Normalized coordinates with format xywh

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        """    
        matched_list = []

        INF = 100000000
        bs, num_queries = outputs["pred_logits"].shape[:2]
        for batch_idx in range(bs):
            cls_pred = outputs["pred_logits"][batch_idx].sigmoid() # https://github.com/JCZ404/Semi-DETR/blob/main/detr_od/models/dense_heads/dino_detr_ssod_head.py#L1110
            bbox_pred = outputs["pred_boxes"][batch_idx]  # [num_queries, 4]
            # We flatten to compute the cost matrices in a batch
            # cls_pred = outputs["pred_logits"].flatten(0, 1).sigmoid() # https://github.com/JCZ404/Semi-DETR/blob/main/detr_od/models/dense_heads/dino_detr_ssod_head.py#L1110
            # bbox_pred = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            gt_labels = targets[batch_idx]["labels"]
            gt_bboxes = targets[batch_idx]["boxes"]
            # gt_labels = torch.cat([v["labels"] for v in targets])
            # gt_bboxes = torch.cat([v["boxes"] for v in targets])
            
            num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
            gt_labels = gt_labels.long()
            # 1. assign -1 by default
            assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            assign_metrics = bbox_pred.new_zeros((num_bboxes, ))

            if num_gts == 0 or num_bboxes == 0:
                # No ground truth or boxes, return empty assignment
                max_overlaps = bbox_pred.new_zeros((num_bboxes, ))
                if num_gts == 0:
                    # No ground truth, assign all to background
                    assigned_gt_inds[:] = 0
            else:
                scores = cls_pred        # [num_bbox, 80]
                overlaps = -generalized_box_iou(box_cxcywh_to_xyxy(bbox_pred),
                                                    box_cxcywh_to_xyxy(gt_bboxes))   # [num_bbox, num_gt]
                bbox_scores = scores[:, gt_labels].detach()
                alignment_metrics = bbox_scores ** alpha * overlaps ** beta # [num_bbox, num_gt]

                _, candidate_idxs = alignment_metrics.topk(                 # [top-k, num_gt]
                        self.candidate_topk, dim=0, largest=True)
                candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gts)]
                is_pos = candidate_metrics > 0

                for gt_idx in range(num_gts):
                    candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
                candidate_idxs = candidate_idxs.view(-1)

                # deal with a single candidate assigned to multiple gt_bboxes
                overlaps_inf = torch.full_like(overlaps,                    # [num_bbox x num_gt,]
                                            -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]            # [num_gt x top-k]
                overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
                overlaps_inf = overlaps_inf.view(num_gts, -1).t()            # [num_gt, num_bbox] -> [num_bbox, num_gt]


                max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

                # assign the background class first
                assigned_gt_inds[:] = 0
                assigned_gt_inds[
                    max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
                assign_metrics[
                    max_overlaps != -INF] = alignment_metrics[max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

                if gt_labels is not None:
                    assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
                    pos_inds = torch.nonzero(
                        assigned_gt_inds > 0, as_tuple=False).squeeze()
                    if pos_inds.numel() > 0:
                        assigned_labels[pos_inds] = gt_labels[
                            assigned_gt_inds[pos_inds] - 1]
                else:
                    assigned_labels = None

            query_idx = torch.arange(num_queries).cuda()
            non_zero_idx = assigned_gt_inds.nonzero(as_tuple=True)[0]
            valid_assigned_gt_inds = assigned_gt_inds[non_zero_idx]
            valid_assigned_gt_inds = valid_assigned_gt_inds - 1 
            valid_query_idx = query_idx[non_zero_idx]

            matched_list.append((valid_query_idx, valid_assigned_gt_inds))

        return matched_list
        # [(tensor([  6,  98, 147, 150, 240]), tensor([4, 3, 2, 0, 1])), (tensor([ 82, 158, 194, 280, 294]), tensor([3, 0, 4, 1, 2]))]
        # tensor([0, 0, 1, 4, 1, 4, 0, 4, 1, 0, 4, 0, 2, 0, 0, 4, 0, 3, 2, 0, 1, 0, 5, 3,
        # 4, 0, 0, 2, 0, 2, 0, 0, 4, 5, 1, 0, 0, 3, 2, 1, 0, 0, 5, 0, 0, 3, 2, 0,
        # 3, 0, 2, 5, 2, 0, 5, 0, 4, 0, 0, 5, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 3, 1,
        # 3, 0, 0, 3, 2, 3, 0, 0, 3, 0, 0, 0, 4, 3, 0, 0, 0, 5, 4, 0, 3, 5, 0, 3,
        # 0, 2, 0, 1])

def build_matcher(args, matcher_type:str = 'hungarian'):
    if matcher_type == 'o2m':
        return O2MMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)
    if matcher_type == 'hungarian':
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)
