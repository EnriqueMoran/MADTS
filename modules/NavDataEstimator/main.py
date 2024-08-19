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
from src.utils.helpers import crop_roi, draw_horizontal_lines, draw_roi


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
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, 
                                              format=self.log_format, 
                                              level=self.log_level, 
                                              config_path=self.config_filepath)
        
        image_size = (1280, 720)
        
        test_img_l = cv2.imread("./modules/NavDataEstimator/test/left.jpg",  cv2.IMREAD_GRAYSCALE)
        test_img_r = cv2.imread("./modules/NavDataEstimator/test/right.jpg", cv2.IMREAD_GRAYSCALE)

        test_img_l = cv2.resize(test_img_l, image_size)
        test_img_r = cv2.resize(test_img_r, image_size)

        with open("./modules/NavDataEstimator/calibration/params/left/calibration.pkl", 'rb') as file:
            camera_matrix_l, dist_l, _, _, obj_points_list_l, img_points_list_l = pickle.load(file)
        
        with open("./modules/NavDataEstimator/calibration/params/right/calibration.pkl", 'rb') as file:
            camera_matrix_r, dist_r, _, _, _, img_points_list_r = pickle.load(file)

        rectified_images = nav_data_estimator.distance_calculator.get_rectified_images(
            image_left=test_img_l, 
            image_right=test_img_r, 
            obj_points_list_l=obj_points_list_l,
            img_points_list_l=img_points_list_l,
            img_points_list_r=img_points_list_r, 
            camera_matrix_l=camera_matrix_l, 
            dist_l=dist_l, 
            camera_matrix_r=camera_matrix_r, 
            dist_r=dist_r
        )
        rectified_left, rectified_right, roi1, _ = rectified_images

        #rectified_left_with_lines = draw_horizontal_lines(image=rectified_left,
        #                                                  line_interval=50,
        #                                                  color=(0, 255, 0),
        #                                                  thickness=2)
        
        #rectified_right_with_lines = draw_horizontal_lines(image=rectified_right,
        #                                                   line_interval=50,
        #                                                   color=(0, 255, 0), 
        #                                                   thickness=2)
        
        #combined_image = cv2.hconcat([rectified_left_with_lines, rectified_right_with_lines])
        #cv2.imshow('Rectified with lines', combined_image)
        #cv2.waitKey(0)

        num_disparities = nav_data_estimator.config_parser.parameters.num_disparities
        block_size = nav_data_estimator.config_parser.parameters.block_size
        depth_map  = nav_data_estimator.distance_calculator.get_depth_map(left_image=rectified_left,
                                                                          right_image=rectified_right,
                                                                          n_disparities=num_disparities,
                                                                          block_size=block_size)
        
        normalized_depth_map = nav_data_estimator.distance_calculator.normalize_depth_map(depth_map)
        normalized_depth_map = crop_roi(normalized_depth_map, roi1)

        h, w = normalized_depth_map.shape[0], normalized_depth_map.shape[1]
        cv2.imshow("Depth map", cv2.resize(normalized_depth_map, (w, h)))
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