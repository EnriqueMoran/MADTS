"""
Implements calibration managemer class.
"""

import cv2
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm

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

        left_calibration  = self.config_parser.left_camera_calibration
        right_calibration = self.config_parser.right_camera_calibration

        img_dir_left           = Path(left_calibration.image_directory).resolve()
        chessboard_width_l     = left_calibration.chessboard_width
        chessboard_height_l    = left_calibration.chessboard_height
        frame_width_l          = left_calibration.frame_width
        frame_height_l         = left_calibration.frame_height
        square_size_l          = left_calibration.chessboard_square_size
        save_calibrated_imgs_l = left_calibration.save_calibration_images
        save_calibrated_path_l = Path(left_calibration.save_calibration_images_path).resolve()
        save_params_file_l     = Path(left_calibration.save_calibration_params_path).resolve()
        load_params_file_l     = Path(left_calibration.load_calibration_params_path).resolve()

        img_dir_right          = Path(right_calibration.image_directory).resolve()
        chessboard_width_r     = right_calibration.chessboard_width
        chessboard_height_r    = right_calibration.chessboard_height
        frame_width_r          = right_calibration.frame_width
        frame_height_r         = right_calibration.frame_height
        square_size_r          = right_calibration.chessboard_square_size
        save_calibrated_imgs_r = right_calibration.save_calibration_images
        save_calibrated_path_r = Path(right_calibration.save_calibration_images_path).resolve()
        save_params_file_r     = Path(right_calibration.save_calibration_params_path).resolve()
        load_params_file_r     = Path(right_calibration.load_calibration_params_path).resolve()

        alpha = self.config_parser.parameters.alpha
        undistort_method = self.config_parser.parameters.undistort_method

        self.left_calibrator = Calibrator(filename=self.log_filepath,
                                          format=self.log_format,
                                          level=self.log_level,
                                          image_dir=img_dir_left,
                                          chessboard_width=chessboard_width_l,
                                          chessboard_height=chessboard_height_l,
                                          frame_width=frame_width_l,
                                          frame_height=frame_height_l,
                                          square_size=square_size_l,
                                          save_calibrated=save_calibrated_imgs_l,
                                          save_calibrated_path=save_calibrated_path_l,
                                          save_param_file=save_params_file_l,
                                          load_param_file=load_params_file_l,
                                          alpha=alpha,
                                          undistort_method=undistort_method)
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
                                           save_calibrated=save_calibrated_imgs_r,
                                           save_calibrated_path=save_calibrated_path_r,
                                           save_param_file=save_params_file_r,
                                           load_param_file=load_params_file_r,
                                           alpha=alpha,
                                           undistort_method=undistort_method)
        self.logger.info(f"Right camera calibrator created.")
    

    def calibrate_cameras_images(self, save_calibrations=True):
        """
        TBD
        """
        self.logger.info(f"Calibrating left camera using images...")
        _, camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = self.left_calibrator.calibrate_camera()

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
        
        self.logger.info(f"Calibrating right camera using images...")
        res_r = self.right_calibrator.calibrate_camera()
        _, camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_list_r, img_points_list_r = res_r

        repr_error_r = self.right_calibrator.get_reprojection_error(obj_points_list_r, 
                                                                    img_points_list_r, 
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
    

    def calibrate_cameras_video2(self, video_path_l, video_path_r, save_calibrations=True):
        """
        TBD
        """
        step = self.config_parser.parameters.video_calibration_step
        self.logger.info(f"Calibrating left camera using video...")
        res_l = self.left_calibrator.calibrate_camera_video(video_path=video_path_l, step=step)
        _, camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = res_l

        repr_error_l = self.left_calibrator.get_reprojection_error(obj_points_list_l,
                                                                   img_points_list_l,
                                                                   rvecs_l, tvecs_l, dist_l,
                                                                   camera_matrix_l)
        self.logger.info(f"Left camera calibrated, reprojection error: {repr_error_l:.16f}")

        self.logger.info(f"Calibrating right camera using video...")
        res_r = self.right_calibrator.calibrate_camera_video(video_path=video_path_r, step=step)
        _, camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_list_r, img_points_list_r = res_r

        repr_error_r = self.right_calibrator.get_reprojection_error(obj_points_list_r, 
                                                                    img_points_list_r, 
                                                                    rvecs_r, tvecs_r, dist_r, 
                                                                    camera_matrix_r)
        self.logger.info(f"Right camera calibrated, reprojection error: {repr_error_r:.16f}")
        
        if len(img_points_list_l) != len(img_points_list_r):
            msg = f"The size of image point list extracted from videos does not match! " +\
                  f"Size of left list: {len(img_points_list_l)} size of right list: " +\
                  f"{len(img_points_list_r)}. The lists will be adjusted to match sizes."
            self.logger.warning(msg)

            min_length = min(len(img_points_list_l), len(img_points_list_r))
            img_points_list_l = img_points_list_l[:min_length]
            img_points_list_r = img_points_list_r[:min_length]
            obj_points_list_l = obj_points_list_l[:min_length]
            obj_points_list_r = obj_points_list_r[:min_length]

        if save_calibrations:
            self.logger.info(f"Saving left camera calibrations...")
            self.left_calibrator.save_calibration(camera_matrix=camera_matrix_l, dist=dist_l,
                                                  rvecs=rvecs_l, tvecs=tvecs_l,
                                                  obj_points=obj_points_list_l,
                                                  img_points=img_points_list_l)
            
            self.logger.info(f"Saving right camera calibrations...")
            self.right_calibrator.save_calibration(camera_matrix=camera_matrix_r, dist=dist_r,
                                                   rvecs=rvecs_r, tvecs=tvecs_r, 
                                                   obj_points=obj_points_list_r,
                                                   img_points=img_points_list_r)
        
        res = {
            'camera_matrix_l': camera_matrix_l,
            'dist_l': dist_l,
            'rvecs_l': rvecs_l,
            'tvecs_l': tvecs_l,
            'obj_points_list_l': obj_points_list_l,
            'img_points_list_l': img_points_list_l,
            
            'camera_matrix_r': camera_matrix_r,
            'dist_r': dist_r,
            'rvecs_r': rvecs_r,
            'tvecs_r': tvecs_r,
            'obj_points_list_r': obj_points_list_r,
            'img_points_list_r': img_points_list_r,
        }
            
        return res
    

    def _get_synchronized_frames(self, video_path_left: Path, video_path_right: Path, 
                                 step: int = 30):
        """
        Extract synchronized frames from two stereo videos with a given step.
        """
        cap_left = cv2.VideoCapture(str(video_path_left))
        cap_right = cv2.VideoCapture(str(video_path_right))

        if not cap_left.isOpened():
            self.logger.error(f"Failed to open video file: {video_path_left}")
            return None

        if not cap_right.isOpened():
            self.logger.error(f"Failed to open video file: {video_path_right}")
            return None
        
        total_frames_left  = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames       = min(total_frames_left, total_frames_right)
        
        frame_idx = 0
        frames_left = []
        frames_right = []

        with tqdm(total=total_frames // step, desc="Extracting synchronized frames") as pbar:
            while cap_left.isOpened() and cap_right.isOpened():
                ret_left, frame_left = cap_left.read()
                ret_right, frame_right = cap_right.read()

                if not ret_left or not ret_right:
                    break

                if frame_idx % step == 0:
                    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

                    frames_left.append(gray_left)
                    frames_right.append(gray_right)
                    pbar.update(1)
                    
                frame_idx += 1
                
                if frame_idx >= total_frames:    # Stop if any video reaches the end
                    break

        cap_left.release()
        cap_right.release()
        return frames_left, frames_right


    def calibrate_cameras_video(self, video_path_l, video_path_r):
        """
        TBD
        """
        self.logger.info(f"Calibrating both cameras using video...")
        frames_left, frames_right = self._get_synchronized_frames(
            video_path_l, video_path_r, self.config_parser.parameters.video_calibration_step
            )

        left_pts, right_pts = [], []
        img_size = None

        chessboard_size = (self.left_calibrator.chessboard_width, 
                           self.left_calibrator.chessboard_height)
         # TODO Remove left, right chessboard size (it must be shared)

        max_iterations = 30       # Same value as used in documentation
        threshold      = 0.001    # Same value as used in documentation
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                max_iterations, threshold)

        frame_pairs = zip(frames_left, frames_right)
        frame_pairs = tqdm(frame_pairs, total=len(frames_left), desc="Processing frames")
        for left_img, right_img in frame_pairs:
            if img_size is None:
                img_size = (left_img.shape[1], left_img.shape[0])
            res_left, corners_left   = cv2.findChessboardCorners(left_img, chessboard_size)
            res_right, corners_right = cv2.findChessboardCorners(right_img, chessboard_size)

            if res_left and res_right:
                corners_left = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1,-1),
                                                termination_criteria)
                corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1,-1), 
                                                termination_criteria)

                left_pts.append(corners_left)
                right_pts.append(corners_right)

        pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
        pattern_points = [pattern_points] * len(left_pts)

        self.logger.info(f"Calling stereoCallibrate with {len(left_pts)} points...")
        err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            pattern_points, left_pts, right_pts, None, None, None, None, img_size, flags=0
        )
        self.logger.info(f"Cameras calibrated!")
        return err, Kl, Dl, Kr, Dr, R, T, E, F


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

        
    