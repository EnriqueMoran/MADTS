"""
TBD
"""
import argparse
import concurrent.futures
import cv2
import numpy as np
import os
import pickle
import sys

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_horizontal_lines, draw_roi, draw_distance


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
        
        video_left  = Path("./modules/NavDataEstimator/calibration/videos/left_camera.mp4")
        video_right = Path("./modules/NavDataEstimator/calibration/videos/right_camera.mp4")
        
        #nav_data_estimator.distance_calculator.calibrate_cameras_video(video_path_l=video_left,
        #                                                               video_path_r=video_right)
        
        image_size = nav_data_estimator.config_parser.parameters.resolution

        cap_left = cv2.VideoCapture(video_left)
        cap_right = cv2.VideoCapture(video_right)

        ret_l, frame_left = cap_left.read()
        ret_r, frame_right = cap_right.read()

        if not ret_l or not ret_r:
            print("Error capturing frames from video.")
            return

        cap_left.release()
        cap_right.release()

        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        test_img_l = cv2.resize(frame_left, image_size)
        test_img_r = cv2.resize(frame_right, image_size)
        
        # test_img_l = cv2.imread("./modules/NavDataEstimator/test/left.jpg",  cv2.IMREAD_GRAYSCALE)
        # test_img_r = cv2.imread("./modules/NavDataEstimator/test/right.jpg", cv2.IMREAD_GRAYSCALE)

        #test_img_l = cv2.resize(test_img_l, image_size)
        #test_img_r = cv2.resize(test_img_r, image_size)

        with open("./modules/NavDataEstimator/calibration/params/calibration_left.pkl", 'rb') as file:
            camera_matrix_l, dist_l, _, _, obj_points_list_l, img_points_list_l = pickle.load(file)
        
        with open("./modules/NavDataEstimator/calibration/params/calibration_right.pkl", 'rb') as file:
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
        #cv2.imshow("Depth map", cv2.resize(normalized_depth_map, (w, h)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        focal_length_l = nav_data_estimator.config_parser.left_camera_specs.focal_length
        pixel_size_l   = nav_data_estimator.config_parser.left_camera_specs.pixel_size

        baseline = nav_data_estimator.config_parser.system_setup.baseline_distance

        distance_map_left = nav_data_estimator.distance_calculator.get_distance_map(depth_map,
                                                                                    focal_length_l,
                                                                                    pixel_size_l,
                                                                                    baseline)
        #points = [(615, 161), (328, 147), (173, 285), (509, 352), (554, 261)]

        margin = 50  
        step = 100
        image_width, image_height = nav_data_estimator.config_parser.parameters.resolution
        x_points = np.arange(margin, image_width - margin, step)
        y_points = np.arange(margin, image_height - margin, step)

        points = [(int(x), int(y)) for x in x_points for y in y_points]

        #draw_distance(image=test_img_l, 
        #              distance_map=distance_map_left,
        #              points=points)

        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_points_list_l, img_points_list_l, img_points_list_r,
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            image_size, flags=cv2.CALIB_FIX_INTRINSIC
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            image_size, R, T, alpha=1
        )

        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
        )

        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
        )

        precomputed_maps = {
            'map_left_x': map_left_x,
            'map_left_y': map_left_y,
            'map_right_x': map_right_x,
            'map_right_y': map_right_y
        }

        cap_left = cv2.VideoCapture(video_left)
        cap_right = cv2.VideoCapture(video_right)

        fps = cap_left.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while cap_left.isOpened() and cap_right.isOpened():
                ret_left, frame_left = cap_left.read()
                ret_right, frame_right = cap_right.read()

                if not ret_left or not ret_right:
                    break

                # Submit the frame processing to the thread pool
                future = executor.submit(nav_data_estimator.distance_calculator.process_frame, 
                                         frame_left, frame_right, nav_data_estimator, 
                                         precomputed_maps, roi1, focal_length_l, pixel_size_l, baseline, points)

                # Get the result
                frame_with_distances = future.result()

                # Display the processed frame in real-time
                cv2.imshow('Real-Time Video with Distances', frame_with_distances)

                # Exit if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_left.release()
        cap_right.release()
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

    args = parser.parse_args([])

    log_filepath   = f"./modules/NavDataEstimator/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format     = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level      = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.run()