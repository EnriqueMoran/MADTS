"""
Display depth map for given image.
"""

import argparse
import cv2
import os
import pickle
import sys

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_depth_map, draw_roi, draw_horizontal_lines


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
        self.log_filepath = log_filepath
        self.log_format   = log_format
        self.log_level    = log_level
        self.logger       = None
        
        self.img_l = None
        self.img_r = None

        self.config_filepath = config_filepath

        log_dir = os.path.dirname(self.log_filepath)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        args_ok = self._check_args(args)
        if not args_ok:
            return
    

    def _check_args(self, args) -> None:
        """
        Check passed args.

        Args:
            - args (argparse.Namespace): Passed args to be checked.
        """
        res = True    # All arguments read successfully

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
            if log_path.parent.exists():
                self.log_filepath = args.log
            else:
                message = f"Warning: logging file directory {log_path} not found. " +\
                          f"Defaulting to {self.log_filepath}."
                print(message)
        
        if not args.keep_logs:
            if os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w'):
                    pass
        
        if args.img_l:
            img_l = Path(args.img_l)
            if img_l.is_file():
                self.img_l = args.img_l
            else:
                res = False
                message = f"Error: File {img_l} not found."
                print(message)
        
        if args.img_r:
            img_r = Path(args.img_r)
            if img_r.is_file():
                self.img_r = args.img_r
            else:
                res = False
                message = f"Error: File {img_r} not found."
                print(message)
        
        return res
    

    def main(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath,
                                              format=self.log_format,
                                              level=self.log_level,
                                              config_path=self.config_filepath)

        params_file_left  = nav_data_estimator.distance_calculator.left_calibrator.load_param_file
        params_file_right = nav_data_estimator.distance_calculator.right_calibrator.load_param_file

        if not params_file_left.exists() or not params_file_right.exists():
            print(f"ERROR: No calibration archives were found, calibration aborted!")
            return

        if not Path(self.img_l).exists() or not Path(self.img_r).exists():
            print(f"ERROR: Images not found!")
            return
        
        self.img_l = cv2.imread(self.img_l, cv2.IMREAD_GRAYSCALE)
        self.img_r = cv2.imread(self.img_r, cv2.IMREAD_GRAYSCALE)

        image_size = nav_data_estimator.config_parser.parameters.resolution

        #self.img_l = cv2.resize(self.img_l, image_size)
        #self.img_r = cv2.resize(self.img_r, image_size)

        with open(params_file_left, 'rb') as file:
            camera_matrix_l, dist_l, _, _, obj_points_list_l, img_points_list_l = pickle.load(file)
        
        with open(params_file_right, 'rb') as file:
            camera_matrix_r, dist_r, _, _, _, img_points_list_r = pickle.load(file)

        camera_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(
            camera_matrix_l, dist_l, image_size, 
            alpha=nav_data_estimator.config_parser.parameters.alpha, centerPrincipalPoint=True
        )
        camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(
            camera_matrix_r, dist_r, image_size, 
            alpha=nav_data_estimator.config_parser.parameters.alpha, centerPrincipalPoint=True
        )

        undistorted_img_l = cv2.undistort(self.img_l, camera_matrix_l, dist_l)
        undistorted_img_r = cv2.undistort(self.img_r, camera_matrix_r, dist_r)

        #undistorted_img_l  = draw_roi(undistorted_img_l, roi_l)
        #undistorted_img_r = draw_roi(undistorted_img_r, roi_r)

        #undistorted_img_l = cv2.resize(undistorted_img_l, (800, 450))
        #undistorted_img_r = cv2.resize(undistorted_img_r, (800, 450))
        #combined_image = cv2.hconcat([undistorted_img_l, undistorted_img_r])
        #cv2.imshow("Undistorted Images", combined_image)
        #cv2.waitKey(0)

        rectified_images = nav_data_estimator.distance_calculator.get_rectified_images(
            image_left=self.img_l, 
            image_right=self.img_r,
            obj_points_list_l=obj_points_list_l,
            img_points_list_l=img_points_list_l,
            img_points_list_r=img_points_list_r,
            camera_matrix_l=camera_matrix_l,
            dist_l=dist_l, 
            camera_matrix_r=camera_matrix_r, 
            dist_r=dist_r
        )

        rectified_left, rectified_right, roi_l, roi_r = rectified_images

        #rectified_left_h  = draw_horizontal_lines(rectified_left)
        #rectified_right_h = draw_horizontal_lines(rectified_right)
        #rectified_left_h  = cv2.resize(rectified_left_h, (800, 450))
        #rectified_right_h = cv2.resize(rectified_right_h, (800, 450))
        #combined_image    = cv2.hconcat([rectified_left_h, rectified_right_h])
        #cv2.imshow("Horizontal lines", combined_image)
        #cv2.waitKey(0)

        #rectified_left  = draw_roi(rectified_left, roi_l)
        #rectified_right = draw_roi(rectified_right, roi_r)    # IMPORTANT:  roi_r should be used!

        #resized_left = cv2.resize(rectified_left, (800, 450))
        #resized_right = cv2.resize(rectified_right, (800, 450))
        #combined_image = cv2.hconcat([resized_left, resized_right])
        #cv2.imshow("Rectified Images", combined_image)
        #cv2.waitKey(0)
        #
        #rectified_left  = crop_roi(rectified_left, roi_l)
        #rectified_right = crop_roi(rectified_right, roi_r)    # IMPORTANT:  roi_r should be used!

        num_disp = nav_data_estimator.config_parser.parameters.num_disparities
        block_size = nav_data_estimator.config_parser.parameters.block_size

        depth_map  = nav_data_estimator.distance_calculator.get_depth_map(
            left_image=rectified_left,
            right_image=rectified_right,
            n_disparities=num_disp,
            block_size=block_size
        )
        
        normalized_depth_map = nav_data_estimator.distance_calculator.normalize_depth_map(depth_map)

        depth_map_and_image = draw_depth_map(rectified_left, normalized_depth_map)
        depth_map_and_image = cv2.resize(depth_map_and_image, (800, 450))

        cv2.imshow("Depth map", depth_map_and_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibrator.")

    parser.add_argument("--level", 
                        type=str,
                        help="Set loggin level. Valid values: DEBUG, INFO, WARN, ERROR, CRITICAL.")

    parser.add_argument("--log",
                        type=str,
                        help="Set logging file path.")

    parser.add_argument("--keep_logs",
                        type=bool,
                        help="If enabled, won't clear logs files on new run.")

    parser.add_argument("--img_l",
                        type=str,
                        help="Left image to get depth map from.")

    parser.add_argument("--img_r",
                        type=str,
                        help="Right image to get depth map from.")

    #args = parser.parse_args()

    args = parser.parse_args(['--img_l', './modules/NavDataEstimator/test/video_3_1_test_left.png',
                              '--img_r', './modules/NavDataEstimator/test/video_3_1_test_right.png',
                              '--level', 'DEBUG',
                              '--log', './modules/NavDataEstimator/logs/20240823.log'])

    log_filepath = f"./modules/NavDataEstimator/logs/log_{datetime.now().strftime('%Y%m%d')}.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.main()