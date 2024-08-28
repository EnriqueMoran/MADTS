"""
Implements calibration managemer class.
"""

import cv2
import numpy as np

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager
from modules.NavDataEstimator.src.distanceEstimator.cameracalibrator import Calibrator
from modules.NavDataEstimator.src.utils.enums import RectificationMode, UndistortMethod
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_distance


__author__ = "EnriqueMoran"


class DistanceCalculator(BaseClass):
    """
    TBD
    
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
        self.calibrator = Calibrator(filename=filename, format=format, level=level,
                                     config_path=config_path)
        self.undistort_method = self.config_parser.parameters.undistort_method

    
    def undistort_image(self, camera_matrix, dist, img):
        """
        TBD
        """
        height, width = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (width, height),
                                                               self.alpha, (width, height))
        
        if self.undistort_method == UndistortMethod.UNDISTORT:
            dst = cv2.undistort(img, camera_matrix, dist, None, new_camera_matrix)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        elif self.undistort_method == UndistortMethod.REMAP:
            mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, 
                                                     new_camera_matrix, (height, width), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst


    def get_depth_map(self, left_image:np.ndarray, right_image:np.ndarray, n_disparities:int=0, 
                      block_size:int=21):
        """
        TBD
        """
        height_l, width_l = left_image.shape[:2]
        height_r, width_r = right_image.shape[:2]

        avg_width  = (width_l + width_r) // 2
        avg_height = (height_l + height_r) // 2

        resized_img_l = cv2.resize(left_image, (avg_width, avg_height))
        resized_img_r = cv2.resize(right_image, (avg_width, avg_height))

        stereo = cv2.StereoBM_create(numDisparities=n_disparities, blockSize=block_size)
        return stereo.compute(resized_img_l, resized_img_r)


    def normalize_depth_map(self, depth_map):
        """
        TBD
        """
        image_copy = depth_map.copy()

        depth_map_normalized = cv2.normalize(image_copy, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = np.uint8(depth_map_normalized)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_VIRIDIS)
        return depth_map_colored


    def get_rectified_images(self, image_left, image_right, obj_points_list_l,
                              img_points_list_l, img_points_list_r, 
                              camera_matrix_l, dist_l, camera_matrix_r, dist_r):
        """
        TBD
        """
        image_size = self.config_parser.parameters.resolution

        flags = (cv2.CALIB_FIX_INTRINSIC | 
                 cv2.CALIB_SAME_FOCAL_LENGTH | 
                 cv2.CALIB_FIX_PRINCIPAL_POINT | 
                 cv2.CALIB_ZERO_TANGENT_DIST
                )

        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_points_list_l, img_points_list_l, img_points_list_r,
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            image_size, flags=flags
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l, dist_l, camera_matrix_r, dist_r, image_size, R, T, 
            alpha=self.config_parser.parameters.alpha
        )

        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
        )

        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
        )

        rectified_left  = cv2.remap(image_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(image_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        kernel = self.config_parser.parameters.gaussian_kernel_size
        rectified_left  = cv2.GaussianBlur(rectified_left, (kernel, kernel), 0)
        rectified_right = cv2.GaussianBlur(rectified_right, (kernel, kernel), 0)

        return rectified_left, rectified_right, roi1, roi2
    

    def get_distance_map(self, depth_map, focal_length, pixel_size, baseline):
        """
        TBD
        """
        focal_length_pixels = (focal_length / pixel_size) * 1000

        disparity_map = np.float32(depth_map)
        disparity_map[disparity_map == 0] = 1e-6

        distance_map = (focal_length_pixels * baseline) / disparity_map
        return distance_map


    def process_frame(self, frame_left, frame_right, nav_data_estimator, precomputed_maps, roi1, 
                      focal_length_l, pixel_size_l, baseline, points):
        frame_left_resized = cv2.resize(frame_left, 
                                        nav_data_estimator.config_parser.parameters.resolution)
        frame_right_resized = cv2.resize(frame_right, 
                                         nav_data_estimator.config_parser.parameters.resolution)

        frame_left_gray = cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2GRAY)
        frame_right_gray = cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2GRAY)

        # Rectify images using precomputed maps
        rectified_left = cv2.remap(frame_left_gray, precomputed_maps['map_left_x'], 
                                   precomputed_maps['map_left_y'], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right_gray, precomputed_maps['map_right_x'], 
                                    precomputed_maps['map_right_y'], cv2.INTER_LINEAR)

        # Depth map calculation
        depth_map = nav_data_estimator.distance_calculator.get_depth_map(
            left_image=rectified_left,
            right_image=rectified_right,
            n_disparities=nav_data_estimator.config_parser.parameters.num_disparities,
            block_size=nav_data_estimator.config_parser.parameters.block_size
        )

        # Normalize and crop the depth map
        normalized_depth_map = nav_data_estimator.distance_calculator.normalize_depth_map(depth_map)
        normalized_depth_map = crop_roi(normalized_depth_map, roi1)

        # Compute distance map
        distance_map_left = nav_data_estimator.distance_calculator.get_distance_map(
            depth_map, focal_length_l, pixel_size_l, baseline
        )

        # Draw distances on the left frame
        frame_with_distances = draw_distance(frame_left_resized, distance_map_left, points)

        return frame_with_distances

        
    