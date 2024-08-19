"""
TBD
"""

import cv2
import numpy as np


__author__ = "EnriqueMoran"


def draw_horizontal_lines(image, line_interval=50, color=(0, 255, 0), thickness=1)->np.array:
    """
    TBD
    Return a copy of the image with horizontal lines.
    """
    image_copy = image.copy()
    height, width = image.shape[:2]
    
    for y in range(0, height, line_interval):
        cv2.line(image_copy, (0, y), (width, y), color, thickness)
    
    return image_copy


def draw_roi(image, roi, color=(0, 255, 0), thickness=2)->np.array:
    """
    TBD
    Return a copy of the image with roi rectangle.
    """ 
    image_copy = image.copy()
    cv2.rectangle(image_copy, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), 
                  color, thickness)
    
    return image_copy

def crop_roi(image, roi)->np.array:
    """
    Return a copy of the image cropped to fit given roi.
    """
    image_copy = image.copy()
    return image_copy[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]