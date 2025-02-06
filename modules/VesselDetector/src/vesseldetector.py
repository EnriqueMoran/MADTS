"""
Implements main class.
"""

import cv2
import numpy as np

from pathlib import Path

from modules.VesselDetector.src.baseclass import BaseClass
from modules.VesselDetector.src.configmanager import ConfigManager
from modules.VesselDetector.src.multicastmanager import MulticastManager
from modules.VesselDetector.src.utils.helpers import NMS


__author__ = "EnriqueMoran"


class VesselDetector(BaseClass):
    """
    This is the main class, its in charge of detecting vessels using a Deep Learning model.

    Args:
        - config_path (str): Path to configuration file.
        - filename (str): Path to store log file; belongs to BaseClass.
        - format (str): Logger format; belongs to BaseClass.
        - level (str): Logger level; belongs to BaseClass.
    """
    def __init__(self, filename:str, format:str, level:str, config_path:str):
        super().__init__(filename, format, level)
        self.config_parser = ConfigManager(filename=filename, format=format, level=level, 
                                           config_path=config_path)
        
        self.multicast_manager = MulticastManager(filename=filename, format=format, level=level, 
                                                  config_path=config_path)
        
        self.model_cfg_path   = Path(self.config_parser.model.model_config_file)
        self.weights_path     = Path(self.config_parser.model.weights_file)
        self.class_names_path = Path(self.config_parser.model.class_names_file)

        self.model_width  = self.config_parser.model.width
        self.model_height = self.config_parser.model.height
        self.class_names = self._get_class_names()
        # TODO: Change to readNetFromONNX
        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg_path , self.weights_path)


    def _get_class_names(self):
        """
        TBD
        """
        class_names = []
        with open (self.class_names_path, 'r') as f:
            class_names = [class_name.strip() for class_name in f.readlines()]
        return class_names


    def get_detections(self, image):
        """
        TBD
        """
        bboxes = []
        class_ids = []
        confidences = []
        min_detection_confidence = 0.1

        factor = 1 / 255
        size   = (self.model_width, self.model_height)
        mean   = (0, 0, 0)
        swap_rb = True
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=factor, size=size, mean=mean, 
                                     swapRB= swap_rb)
        self.net.setInput(blob)

        layer_names   = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)
        detections = [c for out in outs for c in out if c[4] > min_detection_confidence]

        for detection in detections:
            bbox = detection[:4]

            class_id   = np.argmax(detection[5:])
            confidence = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            confidences.append(confidence)
            
        bboxes, class_ids, confidences = NMS(bboxes, class_ids, confidences)

        class_names = [self.class_names[class_id] for class_id in class_ids]
        return bboxes, class_names, confidences


    def get_bboxes_abs(self, img, bboxes):
        """
        Return bboxes with absolute values (not relatives to image size).
        """
        res = []
        for bbox in bboxes:
            x, y, w, h = bbox
            H, W, _ = img.shape
            bbox = [int(x * W), int(y * H), int(w * W), int(h * H)]
            res.append(bbox)
        return res