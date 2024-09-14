import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.VesselDetector.src.utils.helpers import draw, get_outputs, NMS


model_cfg_path = os.path.join("./modules/VesselDetector/models/YOLOv3/cfg/yolov3.cfg")
weights_path = os.path.join("./modules/VesselDetector/models/YOLOv3/weights/yolov3.weights")
class_names_path = os.path.join("./modules/VesselDetector/models/YOLOv3/cfg/classnames.cfg")

test_image_path = os.path.join("C:/Users/Enrik/Downloads/tsdt2.jpg")


# Load class names
with open (class_names_path, 'r') as f:
    class_names = [class_name.strip() for class_name in f.readlines()]


# Load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, weights_path)


# Convert the image
img = cv2.imread(test_image_path)
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)


# Get detections
net.setInput(blob)
detections = get_outputs(net)


# BBoxes, classes, confidences
bboxes = []
class_ids = []
confidences = []

for detection in detections:
    bbox = detection[:4]

    xc, yc, w, h = bbox
    H, W, _ = img.shape
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    class_id = np.argmax(detection[5:])
    confidence = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    confidences.append(confidence)


# Apply NMS
bboxes, class_ids, confidences = NMS(bboxes, class_ids, confidences)
print(class_ids)

for i, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    cv2.putText(img, class_names[class_ids[i]], (int(xc - (w / 2)), int(yc + (h / 2) - 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    
    img = cv2.rectangle(img, (int(xc - (w / 2)), int(yc - (h / 2))), 
                        (int(xc + (w / 2)), int(yc + (h / 2))), (0, 255, 0), 10)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()