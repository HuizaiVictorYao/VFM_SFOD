import torch
import numpy as np


def xywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], dim=-1)


def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def calculate_shannon_entropy(score):
    eps = 1e-9
    score = np.clip(score, eps, 1.0)
    return -np.sum(score * np.log2(score))


def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-9)


@torch.no_grad()
def entropy_weighted_fusion(
    boxes1, scores1, labels1,
    boxes2, scores2, labels2,
    iou_thr=0.7
):
    num_classes = scores1.shape[1]
    device = boxes1.device
    dtype = boxes1.dtype

    # Convert boxes to xyxy
    boxes1_xyxy = xywh_to_xyxy(boxes1).cpu().numpy().tolist()
    boxes2_xyxy = xywh_to_xyxy(boxes2).cpu().numpy().tolist()

    scores1 = scores1.cpu().numpy()
    scores2 = scores2.cpu().numpy()

    all_boxes = boxes1_xyxy + boxes2_xyxy
    all_scores = np.vstack([scores1, scores2])
    num_total = len(all_boxes)

    used = [False] * num_total
    clusters = []

    # Cluster all boxes by IoU (ignoring labels)
    for i in range(num_total):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, num_total):
            if not used[j] and iou(all_boxes[i], all_boxes[j]) > iou_thr:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    fused_boxes, fused_scores, fused_labels = [], [], []

    for cluster in clusters:
        cluster_boxes = np.array([all_boxes[i] for i in cluster])
        cluster_scores = np.array([all_scores[i] for i in cluster])
        entropies = np.array([calculate_shannon_entropy(s) for s in cluster_scores])
        weights = 1.0 / (entropies + 1e-9)
        weights = weights / (weights.sum() + 1e-9)

        fused_box = np.average(cluster_boxes, axis=0, weights=weights)
        fused_score = np.average(cluster_scores, axis=0, weights=weights)

        label = int(np.argmax(fused_score))
        score = fused_score[label]

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_labels.append(label)

    # Convert to tensor
    fused_boxes = torch.tensor(fused_boxes, dtype=dtype)  # xyxy
    fused_scores = torch.tensor(fused_scores, dtype=torch.float32)  # [N, C]
    fused_labels = torch.tensor(fused_labels, dtype=torch.long)
    if len(fused_boxes) == 0:
        fused_boxes = torch.zeros((0, 4), dtype=dtype)
        fused_scores = torch.zeros((0, num_classes), dtype=torch.float32)
        fused_labels = torch.zeros((0,), dtype=torch.long)
    else:
        fused_boxes = torch.tensor(np.array(fused_boxes), dtype=dtype)  # xyxy
        fused_scores = torch.tensor(np.array(fused_scores), dtype=torch.float32)
        fused_labels = torch.tensor(fused_labels, dtype=torch.long)

        fused_boxes = xyxy_to_xywh(fused_boxes)

    fused_boxes = fused_boxes.to(device)
    fused_scores = fused_scores.to(device)
    fused_labels = fused_labels.to(device)

    return fused_boxes, fused_scores, fused_labels