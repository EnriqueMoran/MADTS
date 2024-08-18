"""
TBD
"""
import argparse
import os

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
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, format=self.log_format, 
                                              level=self.log_level, config_path=self.config_filepath)
        
        img_dir_left        = nav_data_estimator.config_parser.left_image_directory
        chessboard_width_l  = nav_data_estimator.config_parser.left_chessboard_width
        chessboard_height_l = nav_data_estimator.config_parser.left_chessboard_height
        frame_width_l       = nav_data_estimator.config_parser.left_frame_width
        frame_height_l      = nav_data_estimator.config_parser.left_frame_height
        square_size_l       = nav_data_estimator.config_parser.left_square_size
        save_calibrated_l   = nav_data_estimator.config_parser.left_save_calibrated
        params_path_l       = nav_data_estimator.config_parser.left_params_directory

        calibrator_left = Calibrator(filename=self.log_filepath, format=self.log_format,
                                     level=self.log_level, image_dir=img_dir_left,
                                     chessboard_width=chessboard_width_l,
                                     chessboard_height=chessboard_height_l,
                                     frame_width=frame_width_l,
                                     frame_height=frame_height_l, square_size=square_size_l,
                                     save_calibrated=save_calibrated_l,
                                     params_path=params_path_l)
        res = calibrator_left.calibrate_camera()
        rms, camera_matrix, dist, rvecs, tvecs, obj_points_list, img_points_list = res

        calibrator_left.save_calibration(camera_matrix=camera_matrix, dist=dist)

        undistorted_img = calibrator_left.undistort_image(camera_matrix, dist,
                                                          Path("./modules/NavDataEstimator/calibration/left/img1.jpg").resolve())
        
        repr_error = calibrator_left.get_reprojection_error(obj_points_list, img_points_list, 
                                                            rvecs, tvecs, dist, camera_matrix)
        print(f"Total error: {repr_error}")
        import cv2
        cv2.imwrite("./modules/NavDataEstimator/test/undistorted1.jpg", undistorted_img)



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