"""
Implements camera calibration for stereo rectification.

The calibration is made using five different photos taken of a chessboard with different
rotations and translations.

Based on: https://github.com/niconielsen32/CameraCalibration/blob/main/calibration.py
"""

import cv2
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager
from modules.NavDataEstimator.src.utils.enums import CalibrationMode


__author__ = "EnriqueMoran"


class Calibrator(BaseClass):
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
        
        calibration = self.config_parser.calibration

        self.save_calibration_images_path = Path(calibration.save_calibration_images_path).resolve()
        self.save_calibration_params_path = Path(calibration.save_calibration_params_path).resolve()
        self.load_calibration_params_path = Path(calibration.load_calibration_params_path).resolve()
        self.calibration_images_dir_right = Path(calibration.calibration_images_dir_right).resolve()
        self.calibration_images_dir_left  = Path(calibration.calibration_images_dir_left).resolve()
        self.save_calibration_images      = calibration.save_calibration_images
        self.save_calibration_params      = calibration.save_calibration_params
        self.calibration_video_right      = Path(calibration.calibration_video_right).resolve()
        self.calibration_video_left       = Path(calibration.calibration_video_left).resolve()
        self.video_calibration_step       = calibration.video_calibration_step
        self.calibration_mode             = calibration.calibration_mode
        self.chessboard_size              = (calibration.chessboard_width, 
                                             calibration.chessboard_height)


    def _calibrate_camera_images(self):
        """
        Calibrate camera using a set of images where a chessboard pattern is present.
        TBD
        Source: https://github.com/niconielsen32/ComputerVision/blob/master/StereoVisionDepthEstimation/stereo_calibration.py
        """
        self.logger.info(f"Calibrating both cameras using images...")

        left_images, right_images = [], []
        left_pts, right_pts       = [], []
        img_size = None    

        left_images.extend(self.calibration_images_dir_left.glob('*.png'))
        left_images.extend(self.calibration_images_dir_left.glob('*.jpg'))
        left_images.extend(self.calibration_images_dir_left.glob('*.jpeg'))
        left_images.extend(self.calibration_images_dir_left.glob('*.bmp'))
        left_images = list(set(left_images))

        right_images.extend(self.calibration_images_dir_right.glob('*.png'))
        right_images.extend(self.calibration_images_dir_right.glob('*.jpg'))
        right_images.extend(self.calibration_images_dir_right.glob('*.jpeg'))
        right_images.extend(self.calibration_images_dir_right.glob('*.bmp'))
        right_images = list(set(right_images))

        if not left_images:
            error_msg = f"No calibration images found in {self.calibration_images_dir_left}!"
            self.logger.error(error_msg)
            return None
        else:
            self.logger.info(f"{len(left_images)} left camera calibration images found.")
        
        if not right_images:
            error_msg = f"No calibration images found in {self.calibration_images_dir_right}!"
            self.logger.error(error_msg)
            return None
        else:
            self.logger.info(f"{len(right_images)} right camera calibration images found.")

        max_iterations = 30       # Same value as used in documentation
        threshold      = 0.001    # Same value as used in documentation
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                max_iterations, threshold)
        
        image_pairs = zip(left_images, right_images)
        image_pairs = tqdm(image_pairs, total=len(left_images), desc="Processing images")
        for idx, left_img, right_img in enumerate(image_pairs):
            if img_size is None:
                img_size = (left_img.shape[1], left_img.shape[0])

            left_img  = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                
            res_left, corners_left   = cv2.findChessboardCorners(left_img, self.chessboard_size)
            res_right, corners_right = cv2.findChessboardCorners(right_img, self.chessboard_size)

            if res_left and res_right:
                corners_left  = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1,-1),
                                                 termination_criteria)
                corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1,-1), 
                                                 termination_criteria)
                left_pts.append(corners_left)
                right_pts.append(corners_right)

                if self.save_calibration_images:
                    save_path = self.save_calibration_images_path
                    save_path = save_path
                    save_path.mkdir(parents=True, exist_ok=True)

                    save_path_left = save_path / f"image_{idx + 1}_left.png"

                    img_rgb = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, self.chessboard_size, corners_left, res_left)
                    cv2.imwrite(str(save_path_left), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path_left}.")

                    save_path_right = save_path / f"image_{idx + 1}_right.png"

                    img_rgb = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, self.chessboard_size, corners_right, 
                                              res_right)
                    cv2.imwrite(str(save_path_right), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path_right}.")
        
        # Create a [(0, 0, 0), (0, 0, 0), ...] matrix and then change to [(1, 0, 0), (2, 0, 0), ...]
        pattern_points = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        pattern_points = [pattern_points] * len(left_pts)
        
        ret_l, matrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(pattern_points, 
                                                                        left_pts, img_size, 
                                                                        None, None)
        height_l, width_l   = left_images[0].shape
        new_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(matrix_l, dist_l, (width_l, height_l), 
                                                            1, (width_l, height_l))

        ret_r, matrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(pattern_points, right_pts, 
                                                                        img_size, None, None)
        height_r, width_r   = right_images[0].shape
        new_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(matrix_r, dist_r, (width_r, height_r), 
                                                            1, (width_r, height_r))


        self.logger.debug(f"Calling stereoCallibrate with {len(left_pts)} points...")
        err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(pattern_points, left_pts, right_pts, 
                                                              new_matrix_l, dist_l, new_matrix_r, 
                                                              dist_r, img_size, flags=0)
        self.logger.info(f"Cameras calibrated!")
        return err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts 
    
    
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
        
        total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_left  = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames       = min(total_frames_left, total_frames_right)
        
        frames_right = []
        frames_left  = []
        frame_idx    = 0

        with tqdm(total=total_frames // step, desc="Synchronizing frames") as pbar:
            while cap_left.isOpened() and cap_right.isOpened():
                ret_left, frame_left   = cap_left.read()
                ret_right, frame_right = cap_right.read()

                if not ret_left or not ret_right:
                    break

                if frame_idx % step == 0:
                    gray_left  = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
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


    def _calibrate_cameras_video(self):
        """
        Calibrate camera using a two videos where a chessboard pattern is present (in both).
        TBD
        Source: https://github.com/niconielsen32/ComputerVision/blob/master/StereoVisionDepthEstimation/stereo_calibration.py
        """
        self.logger.info(f"Calibrating both cameras using video...")
        frames_left, frames_right = self._get_synchronized_frames(self.calibration_video_left,
                                                                  self.calibration_video_right,
                                                                  self.video_calibration_step)

        left_pts, right_pts = [], []
        img_size = None

        max_iterations = 30       # Same value as used in documentation
        threshold      = 0.001    # Same value as used in documentation
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                max_iterations, threshold)

        idx = 0
        frame_pairs = zip(frames_left, frames_right)
        frame_pairs = tqdm(frame_pairs, total=len(frames_left), desc="Processing frames")
        for left_img, right_img in frame_pairs:
            if img_size is None:
                img_size = (left_img.shape[1], left_img.shape[0])
                
            res_left, corners_left   = cv2.findChessboardCorners(left_img, self.chessboard_size)
            res_right, corners_right = cv2.findChessboardCorners(right_img, self.chessboard_size)

            if res_left and res_right:
                corners_left  = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1,-1),
                                                 termination_criteria)
                corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1,-1), 
                                                 termination_criteria)
                left_pts.append(corners_left)
                right_pts.append(corners_right)

                if self.save_calibration_images:
                    save_path = self.save_calibration_images_path
                    save_path = save_path
                    save_path.mkdir(parents=True, exist_ok=True)

                    save_path_left = save_path / f"image_{idx + 1}_left.png"

                    img_rgb = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, self.chessboard_size, corners_left, res_left)
                    cv2.imwrite(str(save_path_left), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path_left}.")

                    save_path_right = save_path / f"image_{idx + 1}_right.png"

                    img_rgb = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, self.chessboard_size, corners_right, 
                                              res_right)
                    cv2.imwrite(str(save_path_right), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path_right}.")
                    idx += 1
        
        # Create a [(0, 0, 0), (0, 0, 0), ...] matrix and then change to [(1, 0, 0), (2, 0, 0), ...]
        pattern_points = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        pattern_points = [pattern_points] * len(left_pts)
        
        ret_l, matrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(pattern_points, 
                                                                        left_pts, img_size, 
                                                                        None, None)
        height_l, width_l   = frames_left[0].shape
        new_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(matrix_l, dist_l, (width_l, height_l), 
                                                            1, (width_l, height_l))

        ret_r, matrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(pattern_points, right_pts, 
                                                                        img_size, None, None)
        height_r, width_r   = frames_right[0].shape
        new_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(matrix_r, dist_r, (width_r, height_r),
                                                            1, (width_r, height_r))


        self.logger.debug(f"Calling stereoCallibrate with {len(left_pts)} points...")
        err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(pattern_points, left_pts, right_pts, 
                                                              new_matrix_l, dist_l, new_matrix_r, 
                                                              dist_r, img_size, flags=0)
        self.logger.info(f"Cameras calibrated!")
        return err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts


    def save_calibration(self, err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, 
                         right_pts):
        """
        TBD
        """
        self.save_calibration_params_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts), 
                    open(self.save_calibration_params_path, "wb" ))
        self.logger.info(f"Calibration params saved as: {self.save_calibration_params_path}.")
    

    def load_calibration(self):
        """
        Load calibration parameters from a pkl file.
        """
        if not self.load_calibration_params_path.exists():
            self.logger.error(f"Calibration file does not exist: {self.load_calibration_params_path}")
            return None

        with open(self.load_calibration_params_path, "rb") as file:
            err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = pickle.load(file)

        self.logger.info(f"Calibration params loaded from: {self.load_calibration_params_path}.")
        res = {
                "err": err,
                "Kl":  Kl,
                "Dl":  Dl,
                "Kr":  Kr,
                "Dr":  Dr,
                "R": R,
                "T": T,
                "E": E,
                "F": F,
                "pattern_points": pattern_points,
                "right_pts": right_pts,
                "left_pts":  left_pts
              }
        return res


    def calibrate_cameras(self):
        """
        TBD
        """
        params = None

        if self.calibration_mode == CalibrationMode.USE_IMAGES:
            left_images  = self.calibration_images_dir_left
            right_images = self.calibration_images_dir_right

            if not left_images.exists() and not any(left_images.iterdir()):
                self.logger.error(f"Left calibration images folder not found or empty. Aborting!")
                return None
            
            if not right_images.exists() and not any(right_images.iterdir()):
                self.logger.error(f"Right calibration images folder not found or empty. Aborting!")
                return None

            params = self._calibrate_camera_images()
        elif self.calibration_mode == CalibrationMode.USE_VIDEOS:
            if not self.calibration_video_left.exists():
                error_msg = f"Calibration video {self.calibration_video_left} not found. Aborting!"
                self.logger.error(error_msg)
                return None
            
            if not self.calibration_video_right.exists():
                error_msg = f"Calibration video {self.calibration_video_right} not found. Aborting!"
                self.logger.error(error_msg)
                return None
            
            params = self._calibrate_cameras_video()

        else:
            self.logger.error(f"Calibration mode {self.calibration_mode} not recognized. Aborting!")
            return None
        
        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = params

        if self.save_calibration:
            self.save_calibration(err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, 
                                  right_pts)
            
        return err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts