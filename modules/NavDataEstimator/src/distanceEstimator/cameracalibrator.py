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
from modules.NavDataEstimator.src.utils.enums import UndistortMethod


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

    def __init__(self, filename:str, format:str, level:str,
                 image_dir:str, chessboard_width:int, chessboard_height:int, frame_width:int, 
                 frame_height:int, square_size:int, save_calibrated:bool, save_calibrated_path:Path,
                 save_param_file:Path, load_param_file:Path, alpha: float, undistort_method:int):
        super().__init__(filename, format, level)
        self.image_dir              = Path(image_dir).resolve()
        self.chessboard_width       = chessboard_width
        self.chessboard_height      = chessboard_height
        self.frame_size             = (frame_width, frame_height)
        self.square_size            = square_size
        self.save_calibrated_imgs   = save_calibrated
        self.save_calibrated_path   = save_calibrated_path
        self.save_param_file        = save_param_file
        self.load_param_file        = load_param_file
        self.alpha                  = alpha
        self.undistort_method       = undistort_method

        self._max_iterations      = 30       # Same value as used in documentation
        self._threshold           = 0.001    # Same value as used in documentation
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                     self._max_iterations, self._threshold)


    def calibrate_camera_images(self):
        """
        Calibrate camera using a set of images where a chessboard pattern is present.
        TBD
        """
        self.logger.info(f"Calibrating camera using images...")

        # Create a [(0, 0, 0), (0, 0, 0), ...] matrix and then change to [(1, 0, 0), (2, 0, 0), ...]
        obj_points = np.zeros((self.chessboard_width * self.chessboard_height, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:self.chessboard_width, 
                                     0:self.chessboard_height].T.reshape(-1, 2)

        obj_points = obj_points * self.square_size

        obj_points_list = []    # 3D points in real world
        img_points_list = []    # 2D points in image plane
        image_list      = []    # List of images for calibration

        image_list.extend(self.image_dir.glob('*.png'))
        image_list.extend(self.image_dir.glob('*.jpg'))
        image_list.extend(self.image_dir.glob('*.jpeg'))
        image_list.extend(self.image_dir.glob('*.bmp'))
        image_list = list(set(image_list))

        if not image_list:
            self.logger.error(f"No calibration images found in {self.image_dir}!")
            return None
        else:
            self.logger.info(f"{len(image_list)} calibration images found.")

        for idx, image in enumerate(image_list):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            chessboard_size = (self.chessboard_width, self.chessboard_height)
            ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)

            if ret:    # If chessboard pattern was found
                obj_points_list.append(obj_points)
                refined_corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                                   self.termination_criteria)
                img_points_list.append(refined_corners)

                if self.save_calibrated_imgs:
                    self.save_calibrated_path  = self.save_calibrated_path
                    self.save_calibrated_path.mkdir(parents=True, exist_ok=True)

                    save_path = self.save_calibrated_path / f"image_{idx + 1}.png"

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, chessboard_size, refined_corners, ret)
                    cv2.imwrite(str(save_path), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path}.")

        flags=cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, 
                                                                     img_points_list, 
                                                                     self.frame_size, None, None,
                                                                     flags)
        
        self.logger.info(f"Camera calibrated! Root Mean Square: {rms}.")
        self.logger.debug(f"Camera matrix: \n{camera_matrix}.")
        self.logger.debug(f"Distortion coeff: {dist}.")
        self.logger.debug(f"Rotation vectors: \n{rvecs}.")
        self.logger.debug(f"Translation vectors: \n{tvecs}.")

        return rms, camera_matrix, dist, rvecs, tvecs, obj_points_list, img_points_list
    

    def calibrate_camera_video(self, video_path:Path, step=30):
        """
        Calibrate camera using frames from a video where a chessboard pattern is present.
        TBD
        """
        self.logger.info(f"Calibrating camera using video with a step of {step} frames...")

        # Create a [(0, 0, 0), (0, 0, 0), ...] matrix and then change to [(1, 0, 0), (2, 0, 0), ...]
        obj_points = np.zeros((self.chessboard_width * self.chessboard_height, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:self.chessboard_width, 
                                     0:self.chessboard_height].T.reshape(-1, 2)
        
        obj_points = obj_points * self.square_size

        obj_points_list = []  # 3D points in real-world space
        img_points_list = []  # 2D points in image plane

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx    = 0
        frame_count  = 1
        with tqdm(total=total_frames // step, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    chessboard_size = (self.chessboard_width, self.chessboard_height)
                    ret, corners    = cv2.findChessboardCorners(gray, chessboard_size, None)

                    if ret:    # If chessboard pattern was found
                        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                                           self.termination_criteria)
                        img_points_list.append(refined_corners)
                        obj_points_list.append(obj_points)

                        if self.save_calibrated_imgs:
                            self.save_calibrated_path.mkdir(parents=True, exist_ok=True)
                            save_path = self.save_calibrated_path / f"image_{frame_count}.jpg"

                            cv2.drawChessboardCorners(frame, chessboard_size, refined_corners, ret)
                            cv2.imwrite(str(save_path), frame)
                            self.logger.info(f"Saved calibration frame as {save_path}.")
                            frame_count += 1
                    pbar.update(1)
                frame_idx += 1
        cap.release()

        if len(obj_points_list) < 1:
            self.logger.error("No valid calibration data was found in the video.")
            return None

        flags=cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, 
                                                                     img_points_list, 
                                                                     gray.shape[::-1], None, None,
                                                                     flags)
        
        self.logger.info(f"Camera calibrated ({frame_count} frames used! Root Mean Square: {rms}.")
        self.logger.debug(f"Camera matrix: \n{camera_matrix}.")
        self.logger.debug(f"Distortion coefficients: {dist}.")
        self.logger.debug(f"Rotation vectors: \n{rvecs}.")
        self.logger.debug(f"Translation vectors: \n{tvecs}.")

        return rms, camera_matrix, dist, rvecs, tvecs, obj_points_list, img_points_list


    def save_calibration(self, camera_matrix, dist, rvecs, tvecs, obj_points, img_points):
        """
        TBD
        """
        self.save_param_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((camera_matrix, dist, rvecs, tvecs, obj_points, img_points), 
                    open(self.save_param_file, "wb" ))
        self.logger.info(f"Calibration params saved as: {self.save_param_file}.")
    

    def load_calibration(self):
        """
        Load calibration parameters from a pkl file.
        """
        if not self.save_param_file.exists():
            self.logger.error(f"Calibration file does not exist: {self.save_param_file}")
            return None

        with open(self.save_param_file, "rb") as file:
            camera_matrix, dist, rvecs, tvecs, obj_points, img_points = pickle.load(file)

        self.logger.info(f"Calibration params loaded from: {self.save_param_file}.")
        return camera_matrix, dist, rvecs, tvecs, obj_points, img_points


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
    

    def get_reprojection_error(self, obj_points_list, img_points_list, rvecs, tvecs, dist, 
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