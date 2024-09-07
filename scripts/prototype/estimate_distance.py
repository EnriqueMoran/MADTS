"""
Receive detected vessel(s) position on picture, estimate its distance and send it to HMI.
"""

import argparse
import concurrent.futures
import cv2
import os
import sys
import time

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_depth_map
from modules.NavDataEstimator.src.utils.globals import MAX_DISPARITY


__author__ = "EnriqueMoran"


class MainApp(BaseClass):
    """
    TBD

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
        else:
            super().__init__(self.log_filepath, self.log_format, self.log_level)

        
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

            file_name, file_extension = os.path.splitext(self.log_filepath)
            comms_in_log  = file_name + "_comms_in" + file_extension
            with open(comms_in_log, 'w'):
                pass

            comms_out_log = file_name + "_comms_out" + file_extension
            with open(comms_out_log, 'w'):
                pass

        return res
    

    def main(self):
        print("Running Distance estimator script...")
        self.logger.info(f"Running Distance estimator script...")
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath,
                                              format=self.log_format,
                                              level=self.log_level,
                                              config_path=self.config_filepath)
        
        params = nav_data_estimator.distance_calculator.calibrator.load_calibration()
        self.logger.info(f"Calibration parameters loaded successfully!")

        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
        params["err"], params["Kl"], params["Dl"], params["Kr"], params["Dr"], params["R"], \
        params["T"], params["E"], params["F"], params["pattern_points"], params["left_pts"], \
        params["right_pts"]

        distance_calculator = nav_data_estimator.distance_calculator
        config_parser = nav_data_estimator.distance_calculator.config_parser

        num_disp   = distance_calculator.config_parser.parameters.num_disparities
        block_size = distance_calculator.config_parser.parameters.block_size
        max_disp   = MAX_DISPARITY

        stream_left  = cv2.VideoCapture(config_parser.stream.left_camera)
        stream_right = cv2.VideoCapture(config_parser.stream.right_camera)
        recording    = None
        
        self.logger.info(f"Waiting for left and right camera streams...")
        while not (stream_left.isOpened() and stream_right.isOpened()):
            if not stream_left.isOpened():
                stream_left = cv2.VideoCapture(config_parser.stream.left_camera)
            if not stream_right.isOpened():
                stream_right = cv2.VideoCapture(config_parser.stream.right_camera)
            cv2.waitKey(100)

        self.logger.info(f"Obtaining frame size...")
        frame_size = None
        while stream_left.isOpened() and stream_right.isOpened():
            # Obtain frame size by reading one frame from both streamings
            ret_right, frame_right = stream_right.read()
            ret_left, frame_left   = stream_left.read()
        
            if not ret_right or not ret_left:
                continue
            
            frame_width, frame_height = frame_left.shape[1], frame_left.shape[0]
            if frame_left.shape != frame_right.shape:
                msg = f"Left frame shape ({frame_left.shape}) and right frame shape " +\
                      f"({frame_right.shape}) doesn't match."
                self.logger.warning(msg)
                frame_width  = min(frame_left.shape[1], frame_right[1])
                frame_height = min(frame_left.shape[0], frame_right[0])
            frame_size = (frame_width, frame_height)
            self.logger.info(f"Frame size: {frame_size}")
            break

        if config_parser.stream.record:
            fps_left  = stream_left.get(cv2.CAP_PROP_FPS)
            fps_right = stream_right.get(cv2.CAP_PROP_FPS)
            fps       = min(fps_left, fps_right)
            codec     = cv2.VideoWriter_fourcc(*'XVID')
            recording = cv2.VideoWriter(config_parser.stream.record_path, codec, fps, frame_size)

        precomputed_params = distance_calculator.precompute_rectification_maps(Kl, Dl, Kr, Dr, 
                                                                               frame_size, R, T)
        
        xmap_l, ymap_l, xmap_r, ymap_r, roi_l, roi_r = precomputed_params["xmap_l"], \
        precomputed_params["ymap_l"], precomputed_params["xmap_r"], precomputed_params["ymap_r"], \
        precomputed_params["roi_l"], precomputed_params["roi_r"]

        self.logger.info(f"Precomputed params loaded!")
        self.logger.debug(f"    xmap_l: {xmap_l}")
        self.logger.debug(f"    ymap_l: {ymap_l}")
        self.logger.debug(f"    xmap_r: {xmap_r}")
        self.logger.debug(f"    ymap_r: {ymap_r}")
        self.logger.debug(f"    roi_l: {roi_l}")
        self.logger.debug(f"    roi_r: {roi_r}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            compute_time = 0
            frame_count  = 0
            while stream_left.isOpened() and stream_right.isOpened():
                ret_right, frame_right = stream_right.read()
                ret_left, frame_left   = stream_left.read()
                frame_count += 1
            
                if not ret_left or not ret_right:
                    break

                if len(frame_left.shape) == 3:
                    frame_left  = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                if len(frame_right.shape) == 3:
                    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

                future_left  = executor.submit(cv2.remap, frame_left, xmap_l, ymap_l, 
                                               cv2.INTER_LINEAR)
                future_right = executor.submit(cv2.remap, frame_right, xmap_r, ymap_r, 
                                               cv2.INTER_LINEAR)
                rect_left, rect_right = future_left.result(), future_right.result()

                start_time_sgbm = time.time()
                stereo_sgbm = cv2.StereoSGBM_create(0, max_disp, block_size, uniquenessRatio=10,
                                                    speckleWindowSize=100, speckleRange=32,
                                                    disp12MaxDiff=1,
                                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                dispmap_sgbm = stereo_sgbm.compute(rect_left, rect_right)
                dispmap_sgbm = distance_calculator.apply_disparity_filter(dispmap_sgbm, stereo_sgbm,
                                                                          rect_left, rect_right)
                elapsed_time_sgbm = time.time() - start_time_sgbm
                compute_time += elapsed_time_sgbm

                if frame_count % 30 == 0:
                    compute_time_avg = compute_time / frame_count
                    self.logger.debug(f"Computation time: {compute_time_avg:.3f} seconds")

                if recording:
                    rect_left_roi    = crop_roi(rect_left, roi_l)
                    dispmap_sgbm_roi = crop_roi(dispmap_sgbm, roi_l)
                    draw_depth_sgbm  = draw_depth_map(rect_left_roi, dispmap_sgbm_roi)
                    recording.write(draw_depth_sgbm)
                
            stream_left.release()
            stream_right.release()
            if recording:
                recording.release()
            self.logger.info(f"Streamings are over!")
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
    
    args = parser.parse_args()

    log_filepath = f"./modules/NavDataEstimator/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format, log_level=log_level, 
                  config_filepath=config_filepath, args=args)

    app.main()