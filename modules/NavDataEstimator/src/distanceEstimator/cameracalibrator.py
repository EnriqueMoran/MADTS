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
from src.baseclass import BaseClass


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
                 frame_height:int, square_size:int, save_calibrated:bool,
                 params_path:str):
        super().__init__(filename, format, level)
        self.image_dir           = Path(image_dir).resolve()
        self.chessboard_width    = chessboard_width
        self.chessboard_height   = chessboard_height
        self.square_size         = square_size
        self.frame_size          = (frame_width, frame_height)
        self.save_calibrated_img = save_calibrated
        self.params_path         = Path(params_path).resolve()
        
        self._max_iterations      = 30
        self._threshold           = 0.001
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                     self._max_iterations, self._threshold)


    def calibrate_camera(self):
        """
        TBD
        """
        self.logger.info(f"Calibrating camera...")
        obj_points = np.zeros((self.chessboard_width  * self.chessboard_height, 3), np.float32)
        # Change from [(0, 0, 0), (0, 0, 0), ...] to [(1, 0, 0), (2, 0, 0), ...]
        obj_points[:,:2] = np.mgrid[0:self.chessboard_width,0:self.chessboard_height].T.reshape(-1,2)

        obj_points = obj_points * self.square_size

        obj_points_list = []    # Store 3D points in real world
        img_points_list = []    # Store 2D points in image plane
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

            if ret:
                obj_points_list.append(obj_points)
                refined_corners = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), self.termination_criteria)
                img_points_list.append(refined_corners)

                if self.save_calibrated_img:
                    save_dir  = self.image_dir / "calibrated"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_path = save_dir / f"image_{idx + 1}.jpg"

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    cv2.drawChessboardCorners(img_rgb, chessboard_size, refined_corners, ret)
                    cv2.imwrite(str(save_path), img_rgb)
                    self.logger.info(f"Saved image with calibration corners to {save_path}.")
        
        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, 
                                                                    img_points_list, 
                                                                    self.frame_size, None, None)
        
        self.logger.info(f"RMS: {rms}.")
        self.logger.info(f"Camera matrix: \n{camera_matrix}.")
        self.logger.info(f"Distortion coeff: {dist}.")
        self.logger.info(f"Rotation vectors: \n{rvecs}.")
        self.logger.info(f"Translation vectors: \n{tvecs}.")

        return rms, camera_matrix, dist, rvecs, tvecs, obj_points_list, img_points_list
    

    def calibrate_camera_video(self, video_path:Path, step=30):
        """
        TBD
        """
        self.logger.info(f"Calibrating camera using video with a step of {step} frames...")

        obj_points = np.zeros((self.chessboard_width * self.chessboard_height, 3), np.float32)
        obj_points[:,:2] = np.mgrid[0:self.chessboard_width, 0:self.chessboard_height].T.reshape(-1, 2)
        obj_points *= self.square_size

        obj_points_list = []  # 3D points in real-world space
        img_points_list = []  # 2D points in image plane

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (1280, 720))
                chessboard_size = (self.chessboard_width, self.chessboard_height)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

                if ret:
                    refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.termination_criteria)
                    img_points_list.append(refined_corners)
                    obj_points_list.append(obj_points)

                    if self.save_calibrated_img:
                        save_dir  = self.params_path / "calibrated"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        save_path = save_dir / f"frame_{frame_idx}.jpg"
                        cv2.drawChessboardCorners(frame, chessboard_size, refined_corners, ret)
                        cv2.imwrite(str(save_path), frame)
                        self.logger.info(f"Saved frame with calibration corners to {save_path}.")

            frame_idx += 1

        cap.release()

        if len(obj_points_list) < 1:
            self.logger.error("No valid calibration data was found in the video.")
            return None

        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, 
                                                                    img_points_list, 
                                                                    gray.shape[::-1], None, None)
        
        self.logger.info(f"RMS: {rms}.")
        self.logger.info(f"Camera matrix: \n{camera_matrix}.")
        self.logger.info(f"Distortion coefficients: {dist}.")
        self.logger.info(f"Rotation vectors: \n{rvecs}.")
        self.logger.info(f"Translation vectors: \n{tvecs}.")

        return rms, camera_matrix, dist, rvecs, tvecs, obj_points_list, img_points_list


    def save_calibration(self, camera_matrix, dist, rvecs, tvecs, obj_points, img_points):
        """
        TBD
        """
        self.params_path.mkdir(parents=True, exist_ok=True)
        pickle.dump((camera_matrix, dist, rvecs, tvecs, obj_points, img_points), 
                    open(self.params_path / "calibration.pkl", "wb" ))
        self.logger.info(f"Calibration params saved as: {self.params_path / 'calibration.pkl'}.")


    def undistort_image(self, camera_matrix, dist, img, method='undistort'):
        """
        TBD
        """
        height, width = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (width,height),
                                                               0.9, (width,height))
        
        if method.lower() == 'undistort':
            dst = cv2.undistort(img, camera_matrix, dist, None, new_camera_matrix)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, 
                                                     new_camera_matrix, (height, width), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
    

    def get_reprojection_error(self, obj_points_list, img_points_list, rvecs, tvecs, dist, camera_matrix):
        """
        TBD
        """
        mean_error = 0

        for i in range(len(obj_points_list)):
            imgpoints2, _ = cv2.projectPoints(obj_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist)
            error = cv2.norm(img_points_list[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error/len(obj_points_list)