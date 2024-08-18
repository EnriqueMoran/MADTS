"""
Implements calibration managemer class.
"""

import os

from src.baseclass import BaseClass
from src.configmanager import ConfigManager
from src.distanceEstimator.cameracalibrator import Calibrator


__author__ = "EnriqueMoran"


class CalibrationManager(BaseClass):
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

    
    def _create_calibrators(self):
        """
        Create and initialize calibrator for each camera.
        """
        self.logger.info(f"Creating camera calibrators...")

        img_dir_left        = self.config_parser.left_image_directory
        chessboard_width_l  = self.config_parser.left_chessboard_width
        chessboard_height_l = self.config_parser.left_chessboard_height
        frame_width_l       = self.config_parser.left_frame_width
        frame_height_l      = self.config_parser.left_frame_height
        square_size_l       = self.config_parser.left_square_size
        save_calibrated_l   = self.config_parser.left_save_calibrated
        params_path_l       = self.config_parser.left_params_directory

        img_dir_right       = self.config_parser.right_image_directory
        chessboard_width_r  = self.config_parser.right_chessboard_width
        chessboard_height_r = self.config_parser.right_chessboard_height
        frame_width_r       = self.config_parser.right_frame_width
        frame_height_r      = self.config_parser.right_frame_height
        square_size_r       = self.config_parser.right_square_size
        save_calibrated_r   = self.config_parser.right_save_calibrated
        params_path_r       = self.config_parser.right_params_directory

        if not os.path.exists(img_dir_left):
            self.logger.warning(f"Image directory for left camera: {img_dir_left} not found!")
        if not os.path.exists(params_path_l):
            self.logger.warning(f"Parameters directory for left camera: {params_path_l} not found!")
        
        if not os.path.exists(img_dir_right):
            self.logger.warning(f"Image directory for right camera: {img_dir_right} not found!")
        if not os.path.exists(params_path_r):
            self.logger.warning(f"Parameters directory for right camera: {params_path_r} not found!")
        
        self.calibrator_left = Calibrator(filename=self.log_filepath, 
                                          format=self.log_format,
                                          level=self.log_level, 
                                          image_dir=img_dir_left,
                                          chessboard_width=chessboard_width_l,
                                          chessboard_height=chessboard_height_l,
                                          frame_width=frame_width_l,
                                          frame_height=frame_height_l,
                                          square_size=square_size_l,
                                          save_calibrated=save_calibrated_l,
                                          params_path=params_path_l)
        self.logger.info(f"Left camera calibrator created.")
        
        self.calibrator_right = Calibrator(filename=self.log_filepath, 
                                           format=self.log_format,
                                           level=self.log_level, 
                                           image_dir=img_dir_right,
                                           chessboard_width=chessboard_width_r,
                                           chessboard_height=chessboard_height_r,
                                           frame_width=frame_width_r,
                                           frame_height=frame_height_r, 
                                           square_size=square_size_r,
                                           save_calibrated=save_calibrated_r,
                                           params_path=params_path_r)
        self.logger.info(f"Right camera calibrator created.")
    

    def calibrate_cameras(self, save_calibrations=True):
        self.logger.info(f"Calibrating left camera...")
        res_l = self.calibrator_left.calibrate_camera()
        rms_l, camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = res_l

        repr_error_l = self.calibrator_left.get_reprojection_error(obj_points_list_l, 
                                                                   img_points_list_l, 
                                                                   rvecs_l, tvecs_l, dist_l, 
                                                                   camera_matrix_l)
        self.logger.info(f"Left camera calibrated, reprojection error: {repr_error_l:.16f}")

        if save_calibrations:
            self.calibrator_left.save_calibration(camera_matrix=camera_matrix_l, dist=dist_l)
        

        res_r = self.calibrator_right.calibrate_camera()
        rms_r, camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_rist_r, img_points_rist_r = res_r

        repr_error_r = self.calibrator_right.get_reprojection_error(obj_points_rist_r, 
                                                                    img_points_rist_r, 
                                                                    rvecs_r, tvecs_r, dist_r, 
                                                                    camera_matrix_r)
        self.logger.info(f"Right camera calibrated, reprojection error: {repr_error_r:.6f}")

        if save_calibrations:
            self.calibrator_right.save_calibration(camera_matrix=camera_matrix_r, dist=dist_r)
        