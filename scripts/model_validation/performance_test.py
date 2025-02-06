import cv2
import numpy as np
import os
import time
import torch

from ultralytics import YOLO


# File paths
image_list_path = ""   # Path to the file containing image paths
label_folder = ""  # Folder containing YOLO format ground truth labels
output_folder = ""    # Folder to save images with drawn boxes and metrics
model_path = "yolov8m.pt"  # Path to the pretrained YOLOv8 model
results_file = os.path.join(output_folder, "results.txt")

# Ensure output directory exists, clear previous results
if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(output_folder)

# Load YOLOv8 model
model = YOLO(model_path)


# Read image paths from train.txt
with open(image_list_path, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

# Initialize performance metrics
total_time = 0
TP, FP, FN = 0, 0, 0  # True Positives, False Positives, False Negatives

# Open results file
with open(results_file, "w") as results_log:
    results_log.write("YOLOv8 Detection Results\n")
    results_log.write("=" * 40 + "\n\n")

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
        print(f"Label file not found for {image_name}, skipping...")
        continue

    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image height and width

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

    # Load image
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image dimensions

    # Draw ground truth boxes (Green)
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = map(int, gt_box)  # Use absolute pixel coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for ground truth
        cv2.putText(image, "GT", (x2 - 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Perform YOLO detection
    start_time = time.time()
    results = model(image_path)
    end_time = time.time()

    frame_time = end_time - start_time
    total_time += frame_time

    pred_boxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  # Get box coordinates & class
            if int(cls.item()) == 8:  # Only keep class 8 (boats)
                pred_boxes.append(box.cpu().numpy())

    # Compare detections with ground truth labels
    matched = set()
    frame_TP, frame_FP, frame_FN = 0, 0, 0  # Track per-frame metrics

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
            TP += 1
            frame_TP += 1
            matched.add(best_match)
        else:
            FP += 1
            frame_FP += 1

        # Draw predicted boxes (Red)
        cv2.rectangle(image, (int(x1_pred), int(y1_pred)), (int(x2_pred), int(y2_pred)), (0, 0, 255), 2)  # Red for predictions
        cv2.putText(image, "Pred", (int(x1_pred), int(y1_pred) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame_FN = len(gt_boxes) - len(matched)  # False Negatives = ground truth objects not detected
    FN += frame_FN

    # Save the annotated image
    output_image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_image_path, image)

    # Log frame results
    with open(results_file, "a") as results_log:
        results_log.write(f"Image: {image_name}\n")
        results_log.write(f"TP: {frame_TP}, FP: {frame_FP}, FN: {frame_FN}\n")
        results_log.write(f"Processing time: {frame_time:.4f} sec\n")
        results_log.write("-" * 40 + "\n")

# Compute final metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0

# Save final metrics to results.txt
with open(results_file, "a") as results_log:
    results_log.write("\nFINAL METRICS\n")
    results_log.write("=" * 40 + "\n")
    results_log.write(f"Precision: {precision:.4f}\n")
    results_log.write(f"Recall: {recall:.4f}\n")
    results_log.write(f"Average detection time per image: {avg_time:.4f} seconds\n")

print("Detection results saved to:", results_file)
