"""
Display depth map for given video.
"""

import argparse
import concurrent.futures
import cv2
import os
import sys
import time

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
        
        if args.stream_l:
            self.stream_l = args.stream_l
            
        else:
            res = False
            message = f"Error: RTMP stream URL for left camera not provided."
            print(message)
        
        if args.stream_r:
            self.stream_r = args.stream_r
        else:
            res = False
            message = f"Error: RTMP stream URL for right camera not provided."
            print(message)
    
        return res
    

    def main(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath,
                                              format=self.log_format,
                                              level=self.log_level,
                                              config_path=self.config_filepath)
        
        distance_calculator = nav_data_estimator.distance_calculator
        
        params = distance_calculator.calibrator.load_calibration()

        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
        params["err"], params["Kl"], params["Dl"], params["Kr"], params["Dr"], params["R"], \
        params["T"], params["E"], params["F"], params["pattern_points"], params["left_pts"], \
        params["right_pts"]

        n_disp     = distance_calculator.config_parser.parameters.num_disparities
        block_size = distance_calculator.config_parser.parameters.block_size
        max_disp   = 128

        cap_l = cv2.VideoCapture(self.stream_l)
        cap_r = cv2.VideoCapture(self.stream_r)

        ret_l, frame_l = cap_l.read()
        _, _ = cap_r.read()    # This is done to synchronize right stream (skip 1 frame)

        scale = 0.7  # Reduce resolution
        new_size = (int(frame_l.shape[1] * scale), int(frame_l.shape[0] * scale))


        output_filename = 'output_depth_map.avi' 
        codec = cv2.VideoWriter_fourcc(*'XVID')  
        fps = 20  
        output_size = (1500, 600) 
        out = cv2.VideoWriter(output_filename, codec, fps, output_size)


        precomputed_params = distance_calculator.precompute_rectification_maps(Kl, Dl, Kr, Dr, 
                                                                               new_size, R, T)
        
        xmap_l, ymap_l, xmap_r, ymap_r, roi_l, roi_r = precomputed_params["xmap_l"], \
        precomputed_params["ymap_l"], precomputed_params["xmap_r"], precomputed_params["ymap_r"], \
        precomputed_params["roi_l"], precomputed_params["roi_r"]

        if not cap_l.isOpened() or not cap_r.isOpened():
            print("Error: Cannot open one of the video files.")
            return
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            while cap_l.isOpened() and cap_r.isOpened():
                ret_l, frame_l = cap_l.read()
                ret_r, frame_r = cap_r.read()

                new_size = (int(frame_l.shape[1] * scale), int(frame_l.shape[0] * scale))

                frame_l = cv2.resize(frame_l, new_size, interpolation=cv2.INTER_LINEAR)
                frame_r = cv2.resize(frame_r, new_size, interpolation=cv2.INTER_LINEAR)

                if not ret_l or not ret_r:
                    break

                if len(frame_l.shape) == 3:
                    frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                if len(frame_r.shape) == 3:
                    frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                future_left  = executor.submit(cv2.remap, frame_l, xmap_l, ymap_l, cv2.INTER_LINEAR)
                future_right = executor.submit(cv2.remap, frame_r, xmap_r, ymap_r, cv2.INTER_LINEAR)
                rect_left, rect_right = future_left.result(), future_right.result()

                display_size = (750, 600)

                start_time_bm = time.time()
                stereo_bm  = cv2.StereoBM_create(n_disp, block_size)
                dispmap_bm = stereo_bm.compute(rect_left, rect_right)
                elapsed_time_bm = time.time() - start_time_bm
                #print(f"StereoBM processing time: {elapsed_time_bm:.2f} seconds")

                start_time_sgbm = time.time()
                stereo_sgbm  = cv2.StereoSGBM_create(0, max_disp, block_size, uniquenessRatio=10,
                                                     speckleWindowSize=100, speckleRange=32,
                                                     disp12MaxDiff=1,
                                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                dispmap_sgbm = stereo_sgbm.compute(rect_left, rect_right)
                elapsed_time_sgbm = time.time() - start_time_sgbm
                #print(f"StereoSGBM processing time: {elapsed_time_sgbm:.2f} seconds")

                dispmap_bm = distance_calculator.apply_disparity_filter(dispmap_bm, stereo_bm,
                                                                        rect_left, rect_right)
        
                dispmap_sgbm = distance_calculator.apply_disparity_filter(dispmap_sgbm, stereo_sgbm,
                                                                          rect_left, rect_right)

                dispmap_bm   = distance_calculator.normalize_depth_map(dispmap_bm)
                dispmap_sgbm = distance_calculator.normalize_depth_map(dispmap_sgbm)

                rect_left  = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
                rect_right = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)

                rect_left = crop_roi(rect_left, roi_l)

                dispmap_bm   = crop_roi(dispmap_bm, roi_l)
                dispmap_sgbm = crop_roi(dispmap_sgbm, roi_l)

                draw_depth_bm   = draw_depth_map(rect_left, dispmap_bm)
                draw_depth_sgbm = draw_depth_map(rect_left, dispmap_sgbm)

                combined_image = cv2.hconcat([cv2.resize(draw_depth_bm, display_size), 
                                              cv2.resize(draw_depth_sgbm, display_size)])
                cv2.imshow('Depth maps', combined_image)
                out.write(combined_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap_l.release()
            cap_r.release()
            out.release()
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

    parser.add_argument("--stream_l",
                        type=str,
                        help="Left camera stream to get depth map from.")

    parser.add_argument("--stream_r",
                        type=str,
                        help="Right camera stream to get depth map from.")

    #args = parser.parse_args()   # TODO Uncomment

    ######################## DEBUG --- MUST BE REMOVED ########################
    args = parser.parse_args(['--stream_l', 'rtmp://192.168.1.140:1935/gopro_left',
                              '--stream_r', 'rtmp://192.168.1.140:1935/gopro_right',
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