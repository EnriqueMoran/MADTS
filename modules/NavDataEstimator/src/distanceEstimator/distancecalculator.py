"""
Implements calibration managemer class.
"""

import cv2
import numpy as np
import os

from pathlib import Path
from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager
from modules.NavDataEstimator.src.distanceEstimator.cameracalibrator import Calibrator
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
        self.log_filepath       = filename
        self.log_format         = format
        self.log_level          = level
        self.left_calibrator    = None
        self.right_calibrator   = None
        self._create_calibrators()

    
    def _create_calibrators(self)->None:
        """
        Create and initialize calibrator for each camera.
        """
        self.logger.info(f"Creating camera calibrators...")

        img_dir_left        = Path(self.config_parser.left_camera_calibration.image_directory).resolve()
        chessboard_width_l  = self.config_parser.left_camera_calibration.chessboard_width
        chessboard_height_l = self.config_parser.left_camera_calibration.chessboard_height
        frame_width_l       = self.config_parser.left_camera_calibration.frame_width
        frame_height_l      = self.config_parser.left_camera_calibration.frame_height
        square_size_l       = self.config_parser.left_camera_calibration.chessboard_square_size
        save_calibrated_l   = self.config_parser.left_camera_calibration.save_calibrated_image
        calibration_img_l   = Path(self.config_parser.left_camera_calibration.save_calibration_images_path).resolve()
        params_file_l       = Path(self.config_parser.left_camera_calibration.save_calibration_params_path).resolve()

        img_dir_right       = Path(self.config_parser.right_camera_calibration.image_directory).resolve()
        chessboard_width_r  = self.config_parser.right_camera_calibration.chessboard_width
        chessboard_height_r = self.config_parser.right_camera_calibration.chessboard_height
        frame_width_r       = self.config_parser.right_camera_calibration.frame_width
        frame_height_r      = self.config_parser.right_camera_calibration.frame_height
        square_size_r       = self.config_parser.right_camera_calibration.chessboard_square_size
        save_calibrated_r   = self.config_parser.right_camera_calibration.save_calibrated_image
        calibration_img_r   = Path(self.config_parser.right_camera_calibration.save_calibration_images_path).resolve()
        params_file_r       = Path(self.config_parser.right_camera_calibration.save_calibration_params_path).resolve()

        if not os.path.exists(img_dir_left):
            msg = f"Image directory for left camera: {img_dir_left} not found!"
            self.logger.warning(msg)
        if not os.path.exists(Path(params_file_l).parent):
            msg = f"Parameters directory for left camera: {Path(params_file_l).parent} not found!"
            self.logger.warning(msg)
        
        if not os.path.exists(img_dir_right):
            msg = f"Image directory for right camera: {img_dir_right} not found!"
            self.logger.warning(msg)
        if not os.path.exists(Path(params_file_r).parent):
            msg = f"Parameters directory for right camera: {Path(params_file_r).parent} not found!"
            self.logger.warning(msg)
        
        self.left_calibrator = Calibrator(filename=self.log_filepath, 
                                          format=self.log_format,
                                          level=self.log_level,
                                          image_dir=img_dir_left,
                                          chessboard_width=chessboard_width_l,
                                          chessboard_height=chessboard_height_l,
                                          frame_width=frame_width_l,
                                          frame_height=frame_height_l,
                                          square_size=square_size_l,
                                          save_calibrated=save_calibrated_l,
                                          calibration_img_path=calibration_img_l,
                                          param_file=params_file_l)
        self.logger.info(f"Left camera calibrator created.")
        
        self.right_calibrator = Calibrator(filename=self.log_filepath, 
                                           format=self.log_format,
                                           level=self.log_level, 
                                           image_dir=img_dir_right,
                                           chessboard_width=chessboard_width_r,
                                           chessboard_height=chessboard_height_r,
                                           frame_width=frame_width_r,
                                           frame_height=frame_height_r, 
                                           square_size=square_size_r,
                                           save_calibrated=save_calibrated_r,
                                           calibration_img_path=calibration_img_r,
                                           param_file=params_file_r)
        self.logger.info(f"Right camera calibrator created.")
    

    def calibrate_cameras(self, save_calibrations=True):
        """
        TBD
        """
        self.logger.info(f"Calibrating left camera...")
        res_l = self.left_calibrator.calibrate_camera()
        _, camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = res_l

        repr_error_l = self.left_calibrator.get_reprojection_error(obj_points_list_l,
                                                                   img_points_list_l,
                                                                   rvecs_l, tvecs_l, dist_l, 
                                                                   camera_matrix_l)
        self.logger.info(f"Left camera calibrated, reprojection error: {repr_error_l:.16f}")

        if save_calibrations:
            self.logger.info(f"Saving left camera calibrations...")
            self.left_calibrator.save_calibration(camera_matrix=camera_matrix_l, dist=dist_l,
                                                  rvecs=rvecs_l,
                                                  tvecs=tvecs_l,
                                                  obj_points=obj_points_list_l,
                                                  img_points=img_points_list_l)
        
        self.logger.info(f"Calibrating right camera...")
        res_r = self.right_calibrator.calibrate_camera()
        _, camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_rist_r, img_points_rist_r = res_r

        repr_error_r = self.right_calibrator.get_reprojection_error(obj_points_rist_r, 
                                                                    img_points_rist_r, 
                                                                    rvecs_r, tvecs_r, dist_r, 
                                                                    camera_matrix_r)
        self.logger.info(f"Right camera calibrated, reprojection error: {repr_error_r:.16f}")

        if save_calibrations:
            self.logger.info(f"Saving right camera calibrations...")
            self.right_calibrator.save_calibration(camera_matrix=camera_matrix_r, dist=dist_r,
                                                  rvecs=rvecs_r,
                                                  tvecs=tvecs_r,
                                                  obj_points=obj_points_list_l,
                                                  img_points=img_points_list_l)
    

    def calibrate_cameras_video(self, video_path_l, video_path_r, save_calibrations=True, step=30):
        """
        TBD
        """
        self.logger.info(f"Calibrating left camera...")
        res_l = self.left_calibrator.calibrate_camera_video(video_path=video_path_l, step=step)
        _, camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = res_l

        repr_error_l = self.left_calibrator.get_reprojection_error(obj_points_list_l,
                                                                   img_points_list_l,
                                                                   rvecs_l, tvecs_l, dist_l, 
                                                                   camera_matrix_l)
        self.logger.info(f"Left camera calibrated, reprojection error: {repr_error_l:.16f}")

        if save_calibrations:
            self.logger.info(f"Saving left camera calibrations...")
            self.left_calibrator.save_calibration(camera_matrix=camera_matrix_l, dist=dist_l,
                                                  rvecs=rvecs_l, tvecs=tvecs_l,
                                                  obj_points=obj_points_list_l,
                                                  img_points=img_points_list_l)
        
        self.logger.info(f"Calibrating right camera...")
        res_r = self.right_calibrator.calibrate_camera_video(video_path=video_path_r, step=step)
        _, camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_rist_r, img_points_rist_r = res_r

        repr_error_r = self.right_calibrator.get_reprojection_error(obj_points_rist_r, 
                                                                    img_points_rist_r, 
                                                                    rvecs_r, tvecs_r, dist_r, 
                                                                    camera_matrix_r)
        self.logger.info(f"Right camera calibrated, reprojection error: {repr_error_r:.16f}")

        if save_calibrations:
            self.logger.info(f"Saving right camera calibrations...")
            self.right_calibrator.save_calibration(camera_matrix=camera_matrix_r, dist=dist_r,
                                                   rvecs=rvecs_r, tvecs=tvecs_r, 
                                                   obj_points=obj_points_rist_r,
                                                   img_points=img_points_rist_r)
    

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

        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_points_list_l, img_points_list_l, img_points_list_r,
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            image_size, flags=cv2.CALIB_FIX_INTRINSIC
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            image_size, R, T, alpha=1
        )

        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
        )

        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
        )

        rectified_left  = cv2.remap(image_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(image_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        rectified_left  = cv2.GaussianBlur(rectified_left, (5, 5), 0)
        rectified_right = cv2.GaussianBlur(rectified_right, (5, 5), 0)

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

        
    