"""
TBD
"""
import argparse
import cv2
import numpy as np
import os
import pickle

from datetime import datetime
from pathlib import Path

from src.navdataestimator import NavDataEstimator
from src.distanceEstimator.cameracalibrator import Calibrator


__author__ = "EnriqueMoran"


class MainApp:
    """
    This class executes the main loop of NavDataEstimator module.

    Args:
        - args (argparse.Namespace): Arguments to update loggers (Main App) parameters. 
        - log_filepath (str): Default path to store Main App messages log file.
        - log_format (str): Main App default logger format.
        - log_level (str): Main App default logger level.
    """

    def __init__(self, log_format:str, log_level:str, log_filepath:str, 
                 config_filepath:str, args:argparse.Namespace):
        self.log_filepath   = log_filepath
        self.log_format     = log_format
        self.log_level      = log_level
        self.logger         = None

        self.config_filepath = config_filepath

        log_dir = os.path.dirname(self.log_filepath)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self._check_args(args)


    def _check_args(self, args) -> None:
        """
        Check passed args.

        Args:
            - args (argparse.Namespace): Passed args to be checked.
        """
        valid_log_levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        if args.level:
            if args.level.upper() in valid_log_levels:
                self.log_level = os.environ.get("LOGLEVEL", args.level.upper())
            else:
                message = f"Warning: logging level {args.level} not found within valid values: " +\
                          f"{valid_log_levels}. Defaulting to {self.log_level}."
                print(message)

        if args.log:
            log_path = Path(args.log)
            if log_path.exists():
                self.log_filepath = args.log
            else:
                message = f"Warning: logging file path {log_path} not found. " +\
                          f"Defaulting to {self.log_filepath}."
                print(message)
        
        if not args.keep_logs:
            if os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w'):
                    pass


    def run(self):
        self.test()


    def test(self):
        def draw_horizontal_lines(image, line_interval=50, color=(0, 0, 255), thickness=1):
            height, width = image.shape[:2]
            
            # Dibujar líneas horizontales espaciadas por 'line_interval' píxeles
            for y in range(0, height, line_interval):
                cv2.line(image, (0, y), (width, y), color, thickness)
            
            return image

        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, format=self.log_format, 
                                              level=self.log_level, config_path=self.config_filepath)
        
        left_video_path  = Path("./modules/NavDataEstimator/calibration/left_images/left_camera.mp4").resolve()
        right_video_path = Path("./modules/NavDataEstimator/calibration/right_images/right_camera.mp4").resolve()
        #nav_data_estimator.distance_calculator.calibrate_cameras_video(left_video_path, 
        #                                                               right_video_path,
        #                                                               step=20)
        

        test_img_l = cv2.imread("./modules/NavDataEstimator/test/left.jpg",  cv2.IMREAD_GRAYSCALE)
        test_img_r = cv2.imread("./modules/NavDataEstimator/test/right.jpg", cv2.IMREAD_GRAYSCALE)

        test_img_l = cv2.resize(test_img_l, (1280, 720))
        test_img_r = cv2.resize(test_img_r, (1280, 720))

        with open("./modules/NavDataEstimator/calibration/params/left/calibration.pkl", 'rb') as file:
            camera_matrix_l, dist_l, rvecs_l, tvecs_l, obj_points_list_l, img_points_list_l = pickle.load(file)
        
        with open("./modules/NavDataEstimator/calibration/params/right/calibration.pkl", 'rb') as file:
            camera_matrix_r, dist_r, rvecs_r, tvecs_r, obj_points_list_r, img_points_list_r = pickle.load(file)

        retval, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_points_list_l, img_points_list_l, img_points_list_r,
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            (1280, 720),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            (1280, 720), R, T, alpha=1
        )

        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_l, dist_l, R1, P1, (1280, 720), cv2.CV_32FC1
        )

        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_r, dist_r, R2, P2, (1280, 720), cv2.CV_32FC1
        )

        rectified_left = cv2.remap(test_img_l, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(test_img_r, map_right_x, map_right_y, cv2.INTER_LINEAR)

        rectified_left_with_lines = draw_horizontal_lines(rectified_left.copy(), line_interval=50, color=(0, 0, 255), thickness=2)
        rectified_right_with_lines = draw_horizontal_lines(rectified_right.copy(), line_interval=50, color=(0, 0, 255), thickness=2)
        combined_image = cv2.hconcat([rectified_left_with_lines, rectified_right_with_lines])

        rectified_left_with_roi = rectified_left.copy()
        rectified_right_with_roi = rectified_right.copy()
        cv2.rectangle(rectified_left_with_roi, (roi1[0], roi1[1]), (roi1[0] + roi1[2], roi1[1] + roi1[3]), (0, 255, 0), 2)
        cv2.rectangle(rectified_right_with_roi, (roi2[0], roi2[1]), (roi2[0] + roi2[2], roi2[1] + roi2[3]), (0, 255, 0), 2)
        cv2.imshow('Rectified Left Image with ROI', rectified_left_with_roi)
        #cv2.imshow('Rectified Right Image with ROI', rectified_right_with_roi)
        #cv2.waitKey(0)


        #cv2.imshow('Rectified Left Image', cv2.resize(rectified_left, (1280, 720)))
        #cv2.imshow('Rectified Right Image', cv2.resize(rectified_right, (1280, 720)))
        #cv2.imshow('Images', cv2.resize(combined_image, (1280, 720)))
        
        undistorted_img_l = nav_data_estimator.distance_calculator.left_calibrator.undistort_image(
            camera_matrix_l, dist_l, test_img_l)
        undistorted_img_r = nav_data_estimator.distance_calculator.right_calibrator.undistort_image(
            camera_matrix_r, dist_r, test_img_r)
        
        #cv2.imshow("undistorted_img_l", cv2.resize(test_img_l, (1280, 720)))
        #cv2.waitKey(0)
        #cv2.imshow("undistorted_img_r map", cv2.resize(test_img_r, (1280, 720)))
        #cv2.waitKey(0)

        depth_map = nav_data_estimator.distance_calculator.get_depth_map(rectified_left,
                                                                         rectified_right,
                                                                         n_disparities=0,
                                                                         block_size=13)

        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

        depth_map_normalized = np.uint8(depth_map_normalized)

        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_VIRIDIS)
        
        cv2.imshow("Depth map", cv2.resize(depth_map_colored, (1280, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADTS Video Synchronizer.")

    parser.add_argument("--level", 
                        type=str,
                        help="Set loggin level. Valid values: DEBUG, INFO, WARN, ERROR, CRITICAL.")

    parser.add_argument("--log", 
                        type=str,
                        help="Set logging file path.")

    parser.add_argument("--keep_logs",
                        type=bool,
                        help="If enabled, won't clear logs files on new run.")

    args = parser.parse_args()

    log_filepath   = f"./modules/NavDataEstimator/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format     = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level      = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.run()