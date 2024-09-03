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
from modules.NavDataEstimator.src.utils.enums import RectificationMode
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_depth_map, \
                                                       draw_horizontal_lines


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
        
        params = nav_data_estimator.distance_calculator.calibrator.load_calibration()

        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
        params["err"], params["Kl"], params["Dl"], params["Kr"], params["Dr"], params["R"], \
        params["T"], params["E"], params["F"], params["pattern_points"], params["left_pts"], \
        params["right_pts"]
        
        image_left  = cv2.imread(self.img_l)
        image_right = cv2.imread(self.img_r)

        n_disp     = nav_data_estimator.distance_calculator.config_parser.parameters.num_disparities
        block_size = nav_data_estimator.distance_calculator.config_parser.parameters.block_size
        max_disp   = 160

        rect_left, rect_right, params = nav_data_estimator.distance_calculator.rectify_images(
            image_left=image_left, image_right=image_right, Kl=Kl, Dl=Dl, Kr=Kr, Dr=Dr, R=R, T=T
        )

        display_size = (800, 600)

        combined_image = cv2.hconcat([cv2.resize(rect_left, display_size), 
                                      cv2.resize(rect_right, display_size)])
        cv2.imshow('Rectified images', combined_image)
        cv2.waitKey(0)

        stereo_bm  = cv2.StereoBM_create(n_disp, block_size)
        dispmap_bm = stereo_bm.compute(rect_left, rect_right)

        stereo_sgbm  = cv2.StereoSGBM_create(0, max_disp, block_size)
        dispmap_sgbm = stereo_sgbm.compute(rect_left, rect_right)

        dispmap_bm   = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_bm)
        dispmap_sgbm = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_sgbm)

        calibration_mode = nav_data_estimator.distance_calculator.rectification_mode

        if calibration_mode == RectificationMode.CALIBRATED_SYSTEM:
            params['xmap'] = params['xmap_l']
            params['ymap'] = params['ymap_l']
        elif calibration_mode == RectificationMode.UNCALIBRATED_SYSTEM:
            params['H'] = params['Hl']
        
        undistorded_bm = nav_data_estimator.distance_calculator.undistort_rectified_image(
            dispmap_bm, **params
        )

        undistorded_sgbm = nav_data_estimator.distance_calculator.undistort_rectified_image(
            dispmap_sgbm, **params
        )

        image_left  = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

        draw_depth_bm   = draw_depth_map(image_left, undistorded_bm)
        draw_depth_sgbm = draw_depth_map(image_left, undistorded_sgbm)

        combined_image = cv2.hconcat([cv2.resize(draw_depth_bm, display_size), 
                                      cv2.resize(draw_depth_sgbm, display_size)])
        cv2.imshow('Depth maps', combined_image)
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

    ######################## DEBUG --- MUST BE REMOVED ########################
    args = parser.parse_args(['--img_l', './modules/NavDataEstimator/test/video_3_1_test_left.png',
                              '--img_r', './modules/NavDataEstimator/test/video_3_1_test_right.png',
                              '--level', 'DEBUG',
                              '--log', './modules/NavDataEstimator/logs/20240823.log'])
    ###########################################################################

    log_filepath = f"./modules/NavDataEstimator/logs/log_{datetime.now().strftime('%Y%m%d')}.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.main()