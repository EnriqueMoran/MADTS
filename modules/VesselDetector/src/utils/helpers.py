"""
TBD
"""

import cv2
import numpy as np


def NMS(bboxes, class_ids, confidences, overlapThresh=0.5):
    """
    TBD
    """
    bboxes = np.asarray(bboxes)
    original_bboxes = bboxes.copy()
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)
    
    if len(bboxes) == 0:
        return [], [], []

    x1 = bboxes[:, 0] - (bboxes[:, 2] / 2)
    y1 = bboxes[:, 1] - (bboxes[:, 3] / 2)
    x2 = bboxes[:, 0] + (bboxes[:, 2] / 2)
    y2 = bboxes[:, 1] + (bboxes[:, 3] / 2)
    transformed_bboxes = np.stack([x1, y1, x2, y2], axis=1)
    
    idxs = np.argsort(confidences)[::-1]
    
    picked_bboxes = []
    picked_class_ids = []
    picked_confidences = []
    
    while len(idxs) > 0:
        current = idxs[0]
        picked_bboxes.append(original_bboxes[current])
        picked_class_ids.append(class_ids[current])
        picked_confidences.append(confidences[current])
        
        rest_idxs = idxs[1:]
        suppress_idxs = []
        
        for idx in rest_idxs:
            iou = compute_iou(transformed_bboxes[current], transformed_bboxes[idx])
            if iou > overlapThresh:
                suppress_idxs.append(idx)
        
        idxs = np.setdiff1d(rest_idxs, suppress_idxs)
    
    return np.array(picked_bboxes), np.array(picked_class_ids), np.array(picked_confidences)


def compute_iou(box1, box2):
    """
    TBD
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_width  = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area   = inter_width * inter_height

    box1_area  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def draw(bbox, img):
    x, y, width, height = bbox
    top_left = (int(x - width / 2), int(y - height / 2))
    bottom_right = (int(x + width / 2), int(y + height / 2))

    color=(0, 255, 0)
    thickness=2

    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img