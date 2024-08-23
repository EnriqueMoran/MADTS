"""
Calibrate cameras using two video sources and store the parameters in specified file paths.
"""

import argparse
import os
import sys

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator


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
        self.log_filepath = log_filepath
        self.log_format   = log_format
        self.log_level    = log_level
        self.logger       = None

        self.video_left  = None
        self.video_right = None

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
        
        if args.video_l:
            video_l = Path(args.video_l)
            if video_l.is_file():
                self.video_left = args.video_l
            else:
                res = False
                message = f"Error: Left camera file {video_l} not found."
                print(message)
        
        if args.video_r:
            video_r = Path(args.video_r)
            if video_r.is_file():
                self.video_right = args.video_r
            else:
                res = False
                message = f"Error: Right camera file {video_r} not found."
                print(message)

        return res


    def main(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath,
                                              format=self.log_format,
                                              level=self.log_level,
                                              config_path=self.config_filepath)
        
        cl = nav_data_estimator.distance_calculator.left_calibrator.calibration_image_path.exists()
        cr = nav_data_estimator.distance_calculator.right_calibrator.calibration_image_path.exists()
        if not self.video_left and not self.video_right and not cl and not cr:
            print(f"ERROR: No calibration videos nor calibration images directories were found, " +\
                  f"calibration aborted!")
            return
        
        if self.video_left and self.video_right:    # Calibrate using video
            print("Calibrating cameras using the following videos:")
            print(f"{self.video_left}\n{self.video_right}")

            nav_data_estimator.distance_calculator.calibrate_cameras_video(
                video_path_l=self.video_left,
                video_path_r=self.video_right,
                save_calibrations=nav_data_estimator.config_parser.left_camera_calibration.save_calibration_params,
                step=nav_data_estimator.config_parser.parameters.video_calibration_step
                )
        
        elif not self.video_left and not self.video_right:    # Calibrate using images
            print("Calibrating cameras using images...")

            nav_data_estimator.distance_calculator.calibrate_cameras(
                 save_calibrations=nav_data_estimator.config_parser.right_camera_calibration.save_calibration_params
                 )
        
        else:
            print("ERROR: only one calibration video found!")
            print(f"Left video: {self.video_left}")
            print(f"Right video: {self.video_right}")
            return
        
        print(f"Cameras calibrated. \nParameters saved as " +\
              f"{nav_data_estimator.distance_calculator.left_calibrator.param_file} and " +\
              f"{nav_data_estimator.distance_calculator.right_calibrator.param_file}")
        
        return


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
    
    parser.add_argument("--video_l",
                        type=str,
                        help="Left camera calibration video.")
    
    parser.add_argument("--video_r",
                        type=str,
                        help="Right camera calibration video.")

    #args = parser.parse_args()

    args = parser.parse_args(['--video_l', './modules/NavDataEstimator/calibration/videos/left_camera.mp4',
                              '--video_r', './modules/NavDataEstimator/calibration/videos/right_camera.mp4',
                              '--level', 'DEBUG',
                              '--log', './modules/NavDataEstimator/logs/20240823.log'])

    log_filepath = f"./modules/NavDataEstimator/logs/log_{datetime.now().strftime('%Y%m%d')}.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.main()