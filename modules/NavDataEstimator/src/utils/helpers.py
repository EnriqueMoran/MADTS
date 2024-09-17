"""
TBD
"""

import cv2
import math
import numpy as np


__author__ = "EnriqueMoran"


def draw_horizontal_lines(image, line_interval=50, color=(0, 255, 0), thickness=1)->np.array:
    """
    TBD
    Return a copy of the image with horizontal lines.
    """
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
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
    # Check if the image is grayscale (single channel) or color (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
        # TODO Fix, check also distance map
        # TODO apply distance to all

    for (x, y) in points:
        distance = distance_map[y, x]

        if isinstance(distance, np.ndarray):
            distance = distance[0]

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
    
    return image


def draw_depth_map(img, depth_map):
    """
    Return depth map over given image.
    """
    height, width = depth_map.shape[:2]
    img = cv2.resize(img, (width, height))

    # If the image is in grayscale, convert it to BGR
    if len(img.shape) == 2:  # Grayscale image
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img

    alpha = 0.4  # Transparency factor for the left image
    beta = 1.0 - alpha  # Transparency factor for the depth map
    combined_image = cv2.addWeighted(img_color, alpha, depth_map, beta, 0)
    return combined_image


def get_reprojection_error(obj_points_list, img_points_list, rvecs, tvecs, dist, 
                           camera_matrix):
        """
        TBD
        """
        mean_error = 0

        for i in range(len(obj_points_list)):
            img_points_2, _ = cv2.projectPoints(obj_points_list[i], rvecs[i], tvecs[i], 
                                                camera_matrix, dist)
            error = cv2.norm(img_points_list[i], img_points_2, cv2.NORM_L2) / len(img_points_2)
            mean_error += error

        return mean_error / len(obj_points_list)


def draw_epipolar_lines(image_left, image_right, lines, pts_left, pts_right):
    """
    TBD
    """
    r, c = image_left.shape
    image_left = cv2.cvtColor(image_left, cv2.COLOR_GRAY2BGR)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_GRAY2BGR)
    
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts_left, pts_right):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        image_left = cv2.line(image_left, (x0, y0), (x1, y1), color, 1)
        image_left = cv2.circle(image_left, tuple(pt1), 5, color, -1)
        image_right = cv2.circle(image_right, tuple(pt2), 5, color, -1)
    return image_left, image_right


def draw_distance_cloud(image, points_with_distances, detection_kernel_size):
    """
    Draws red or green circles at the specified positions on the image and displays
    the distances from the provided points_with_distances dictionary. Also highlights 
    the area around the points based on the detection kernel size with transparency.
    
    Args:
        image: The input image where distances will be drawn.
        points_with_distances: Dictionary with (x, y) as keys and distances as values.
        detection_kernel_size: The size of the kernel to determine the area to be highlighted.
        
    Returns:
        image: The image with distances and areas drawn.
    """
    overlay = image.copy()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  

    kernel_half = detection_kernel_size // 2

    for (x, y), distance in points_with_distances.items():
        if math.isnan(distance):
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            text = "Nan"
            # Adjusting the text position to be just above the kernel area
            text_x, text_y = x - kernel_half, y - kernel_half - 10
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)
        else:
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            text = f"{distance:.2f} cm"
            # Adjusting the text position to be just above the kernel area
            text_x, text_y = x - kernel_half, y - kernel_half - 10
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, (0, 255, 0), 2)
            
            x_min = max(x - kernel_half, 0)
            y_min = max(y - kernel_half, 0)
            x_max = min(x + kernel_half, image.shape[1] - 1)
            y_max = min(y + kernel_half, image.shape[0] - 1)

            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)

    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image
