"""
Implements calibration managemer class.
"""

import cv2
import numpy as np
import os

from src.baseclass import BaseClass
from src.configmanager import ConfigManager
from src.distanceEstimator.cameracalibrator import Calibrator


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
        self.log_filepath     = filename
        self.log_format       = format
        self.log_level        = level
        self.left_calibrator  = None
        self.right_calibrator = None
        self._create_calibrators()

    
    def _create_calibrators(self)->None:
        """
        Create and initialize calibrator for each camera.
        """
        self.logger.info(f"Creating camera calibrators...")

        img_dir_left        = self.config_parser.left_camera_calibration.image_directory
        chessboard_width_l  = self.config_parser.left_camera_calibration.chessboard_width
        chessboard_height_l = self.config_parser.left_camera_calibration.chessboard_height
        frame_width_l      = self.config_parser.left_camera_calibration.frame_width
        frame_height_l     = self.config_parser.left_camera_calibration.frame_height
        square_size_l      = self.config_parser.left_camera_calibration.chessboard_square_size
        save_calibrated_l  = self.config_parser.left_camera_calibration.save_calibrated_image
        params_dir_l       = self.config_parser.left_camera_calibration.save_calibration_params_path

        img_dir_right       = self.config_parser.right_camera_calibration.image_directory
        chessboard_width_r  = self.config_parser.right_camera_calibration.chessboard_width
        chessboard_height_r = self.config_parser.right_camera_calibration.chessboard_height
        frame_width_r     = self.config_parser.right_camera_calibration.frame_width
        frame_height_r    = self.config_parser.right_camera_calibration.frame_height
        square_size_r     = self.config_parser.right_camera_calibration.chessboard_square_size
        save_calibrated_r = self.config_parser.right_camera_calibration.save_calibrated_image
        params_dir_r      = self.config_parser.right_camera_calibration.save_calibration_params_path

        if not os.path.exists(img_dir_left):
            msg = f"Image directory for left camera: {img_dir_left} not found!"
            self.logger.warning(msg)
        if not os.path.exists(params_dir_l):
            msg = f"Parameters directory for left camera: {params_dir_l} not found!"
            self.logger.warning(msg)
        
        if not os.path.exists(img_dir_right):
            msg = f"Image directory for right camera: {img_dir_right} not found!"
            self.logger.warning(msg)
        if not os.path.exists(params_dir_r):
            msg = f"Parameters directory for right camera: {params_dir_r} not found!"
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
                                          params_path=params_dir_l)
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
                                           params_path=params_dir_r)
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
            self.left_calibrator.save_calibration(camera_matrix=camera_matrix_l, dist=dist_l)
        
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
            self.right_calibrator.save_calibration(camera_matrix=camera_matrix_r, dist=dist_r)
    

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
        image_size = (1280, 720)
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
    