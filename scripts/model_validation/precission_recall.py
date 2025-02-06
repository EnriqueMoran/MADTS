import cv2
import numpy as np
import os
import time
import torch

from ultralytics import YOLO

# File paths
image_list_path = ""
label_folder = ""
output_folder = ""
model_path = "yolov8m.pt"
results_file = os.path.join(output_folder, "precision_recall_results.txt")

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load YOLOv8 model
model = YOLO(model_path)

# Read image paths from validation.txt
with open(image_list_path, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

# Confidence thresholds to test
#thresholds = np.linspace(0.1, 0.9, 9)  # Values from 0.1 to 0.9
thresholds = np.arange(0.1, 1.5, 0.05)  # Generates [0.05, 0.10, 0.15, ..., 0.95]


# Function to compute Intersection over Union (IoU)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Store results for each confidence threshold
precision_recall_data = []

# Iterate over confidence thresholds
for conf_thresh in thresholds:
    total_TP, total_FP, total_FN = 0, 0, 0  # Reset counts

    # Iterate over all images
    for image_path in image_paths:
        image_name = os.path.basename(image_path)

        label_path_jpg = os.path.join(label_folder, image_name.replace(".jpg", ".txt"))
        label_path_png = os.path.join(label_folder, image_name.replace(".png", ".txt"))

        if os.path.exists(label_path_jpg):
            label_path = label_path_jpg
        elif os.path.exists(label_path_png):
            label_path = label_path_png
        else:
            continue  # Skip images without labels

        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # Read ground truth labels
        with open(label_path, "r") as f:
            gt_boxes = []
            for line in f:
                data = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, data)
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                gt_boxes.append([x1, y1, x2, y2])

        # Perform YOLO detection with confidence threshold
        results = model(image_path, conf=conf_thresh)

        pred_boxes = []
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                if int(cls.item()) == 8 and conf.item() >= conf_thresh:  # Only keep class 8 (boats)
                    pred_boxes.append(box.cpu().numpy())

        # Compare detections with ground truth labels
        matched = set()
        for pred_box in pred_boxes:
            x1_pred, y1_pred, x2_pred, y2_pred = pred_box
            best_iou = 0
            best_match = None
            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

            if best_iou >= 0.5 and best_match is not None:
                total_TP += 1
                matched.add(best_match)
            else:
                total_FP += 1

        total_FN += len(gt_boxes) - len(matched)  # False Negatives = ground truth objects not detected

    # Compute Precision and Recall
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    precision_recall_data.append((conf_thresh, precision, recall))

# Save Precision-Recall data
with open(results_file, "w") as results_log:
    results_log.write("Confidence Threshold | Precision | Recall\n")
    results_log.write("=" * 40 + "\n")
    for conf, prec, rec in precision_recall_data:
        results_log.write(f"{conf:.2f} | {prec:.4f} | {rec:.4f}\n")

print("Precision-Recall results saved to:", results_file)
