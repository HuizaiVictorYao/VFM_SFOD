import torch

def bbox_iou(bbox1, bbox2):
    """
    fuction: calculate IoU between two sets of bbox
    input: bbox1 and bbox2, both with shape [m, 4]
    return: IoU matrix with shape [m,m]
    """
    b1_x1, b1_y1 = bbox1[:, 0], bbox1[:, 1]
    b1_x2, b1_y2 = bbox1[:, 0] + bbox1[:, 2], bbox1[:, 1] + bbox1[:, 3]
    b2_x1, b2_y1 = bbox2[:, 0], bbox2[:, 1]
    b2_x2, b2_y2 = bbox2[:, 0] + bbox2[:, 2], bbox2[:, 1] + bbox2[:, 3]

    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area)
    return iou

def match_bboxes(b1, b2):
    """
    For each bbox in b1, finding one index in b2, which belongs to the box with highest IoU with current box in b1

    input:    b1, b2, [b, m, 4]
    return:   tensor of [b,m], representing matched idx

    example:
        b1 = torch.tensor([
            [[50, 50, 100, 100], [150, 150, 200, 200], [200, 200, 50, 50]],
            [[30, 30, 60, 60], [100, 100, 80, 80], [150, 150, 120, 120]]
        ])

        b2 = torch.tensor([
            [[200, 200, 50, 50], [50, 50, 100, 100], [150, 150, 200, 200]],
            [[30, 30, 60, 60], [150, 150, 120, 120], [100, 100, 80, 80]]
        ])
    output:
        tensor([[1, 2, 0],
            [0, 2, 1]])
    """
    b, m, _ = b1.shape
    match_indices = torch.zeros(b, m, dtype=torch.long)

    for i in range(b):
        iou_matrix = bbox_iou(b1[i], b2[i])
        match_indices[i] = torch.argmax(iou_matrix, dim=1)

    return match_indices

def rearrange_tensor(tensor, indices):
    b, m, c = tensor.shape
    rearranged_tensor = tensor[torch.arange(b).unsqueeze(1), indices]
    return rearranged_tensor

if __name__ == "__main__":
    # 示例数据
    b1 = torch.tensor([
        [[50, 50, 100, 100], [150, 150, 200, 200], [200, 200, 50, 50]],
        [[30, 30, 60, 60], [100, 100, 80, 80], [150, 150, 120, 120]]
    ])

    b2 = torch.tensor([
        [[200, 200, 50, 50], [50, 50, 100, 100], [150, 150, 200, 200]],
        [[30, 30, 60, 60], [150, 150, 120, 120], [100, 100, 80, 80]]
    ])
    # b2 = torch.tensor([
    #     [[60, 60, 90, 90], [140, 140, 180, 180], [210, 210, 40, 40]],
    #     [[35, 35, 55, 55], [110, 110, 70, 70], [160, 160, 110, 110]]
    # ])

    matched_indices = match_bboxes(b1, b2)
    print(matched_indices)
    print(rearrange_tensor(b2, matched_indices))

