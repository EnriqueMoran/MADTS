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


def draw_distance(image, distance_map, points):
    """
    Draws red circles at the specified positions on the image and displays
    the distances from the distance map.
    
    Args:
       TBD
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (x, y) in points:
        distance = distance_map[y, x]

        if distance < 0:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            text = "Distance not available"
            cv2.putText(image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            text = f"{distance:.2f} cm"
            cv2.putText(image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, (0, 255, 0), 2)
    
    cv2.imshow('Distance Map', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()