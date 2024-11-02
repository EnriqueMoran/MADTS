"""
Receive detected vessel(s) position on picture, estimate its distance and send it to HMI.
"""

import argparse
import concurrent.futures
import cv2
import os
import sys
import threading
import time

from datetime import datetime
from math import isnan, sqrt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.common.comms.navdata import NavData
from modules.common.streamconsumer import StreamConsumer
from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_depth_map, draw_distance_cloud


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
        self.log_filepath   = log_filepath
        self.log_format     = log_format
        self.log_level      = log_level
        self.logger         = None
        self.last_frame     = None
        self.stop_event     = threading.Event()
        self.display_thread = None

        self.config_filepath = config_filepath

        log_dir = os.path.dirname(self.log_filepath)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        args_ok = self._check_args(args)
        if not args_ok:
            return
        else:
            super().__init__(self.log_filepath, self.log_format, self.log_level)
    

    def _start_display_thread(self, window_size):
        """
        TBD
        """
        self.display_thread = threading.Thread(target=self._display_video, args=(window_size,))
        self.display_thread.start()


    def _stop_display_thread(self):
        """
        TBD
        """
        self.stop_event.set()
        if self.display_thread is not None:
            self.display_thread.join()
        
    
    def _display_video(self, params):
        """
        TBD
        """
        while not self.stop_event.is_set():
            if self.last_frame is not None:
                cv2.imshow('Distance map', cv2.resize(self.last_frame, params[0]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()

        
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


    def _get_frame_size(self, stream_left, stream_right):
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
            return frame_size

        return None


    def _get_streams(self, config_parser):
        stream_left  = StreamConsumer(config_parser.stream.left_camera)
        stream_right = StreamConsumer(config_parser.stream.right_camera)

        # Wait until streams are available
        while not (stream_left.isOpened() and stream_right.isOpened()):
            if not stream_left.isOpened():
                stream_left.release()
                stream_left  = StreamConsumer(config_parser.stream.left_camera)
            if not stream_right.isOpened():
                stream_right.release()
                stream_right = StreamConsumer(config_parser.stream.right_camera)
            time.sleep(1)

        return stream_left, stream_right


    def _get_distance_detections(self, detection_1, detection_2):
        x_dist = abs(detection_2.x - detection_1.x)
        y_dist = abs(detection_2.y - detection_1.y)
        return sqrt(x_dist ** 2 + y_dist ** 2)


    def _correlate_detections(self, detections, threshold):
        """
        Only for prototype version: discard detections that are too close, meaning that both refers
        to the same detected object.
        """
        res = []
        for detection in detections:
            discard_detection = False
            for j in range(len(res)):
                if self._get_distance_detections(detection, res[j]) < threshold:
                    discard_detection = True
                    break
            if not discard_detection:
                res.append(detection)
        return res
    

    def _get_recording_frame(self, distance_calculator, frame_left, aligned_map, 
                             detection_list, kernel_size, frame_count=0):
        """
        Function to create frame for video recording in a separate thread.
        
        Args:
            - recording: The VideoWriter object for recording the video.
            - config_parser: The configuration parser with stream settings.
            - frame_left: The current left frame.
            - aligned_map: The disparity map aligned with the left frame.
            - frame_size: The size of the frame.
            - detection_list: List of detections in the current frame.
        """
        aligned_norm = distance_calculator.normalize_depth_map(aligned_map)
        draw_depth_sgbm = draw_depth_map(frame_left, aligned_norm)

        dist_map = draw_distance_cloud(draw_depth_sgbm, detection_list, kernel_size)

        ####################################### DEBUG DELETE #######################################
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(dist_map, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255), 2, cv2.LINE_AA) 
        cv2.putText(dist_map, f"Time: {current_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0, 0, 255), 2, cv2.LINE_AA)
        ############################################################################################
        
        return dist_map


    def main(self):
        info_msg = f"Running Distance estimator script..."
        print(info_msg)
        self.logger.info(info_msg)

        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, format=self.log_format,
                                              level=self.log_level,
                                              config_path=self.config_filepath)
        
        params = nav_data_estimator.distance_calculator.calibrator.load_calibration()
        info_msg = f"Calibration parameters loaded successfully!"
        print(info_msg)
        self.logger.info(info_msg)

        Kl, Dl = params["Kl"], params["Dl"]
        Kr, Dr = params["Kr"], params["Dr"]
        R, T   = params["R"], params["T"]

        distance_calculator = nav_data_estimator.distance_calculator
        config_parser = nav_data_estimator.distance_calculator.config_parser
        scale = nav_data_estimator.config_parser.stream.scale
        

        if config_parser.stream.show_video:
            window_size = (config_parser.stream.video_width, 
                           config_parser.stream.video_height)
            self._start_display_thread((window_size, ))

        nav_data_estimator.multicast_manager.start_communications()

        info_msg = f"Waiting for left and right camera streams..."
        print(info_msg)
        self.logger.info(info_msg)

        get_stream_start = time.time()
        stream_left, stream_right = self._get_streams(config_parser)
        
        get_stream_elapsed = time.time() - get_stream_start
        self.logger.debug(f"Time required to connect to streams: {get_stream_elapsed:.2f} secs.")
        
        self.logger.info(f"Obtaining frame size...")

        frame_size = self._get_frame_size(stream_left, stream_right)
        frame_size = (int(frame_size[0] * scale), int(frame_size[1] * scale))
        
        precomputed_params = distance_calculator.precompute_rectification_maps(Kl, Dl, Kr, Dr, 
                                                                               frame_size, R, T)
        
        xmap_l, ymap_l, xmap_r, ymap_r, roi_l, Q = precomputed_params["xmap_l"], \
        precomputed_params["ymap_l"], precomputed_params["xmap_r"], precomputed_params["ymap_r"], \
        precomputed_params["roi_l"], precomputed_params["Q"]

        self.logger.info(f"Precomputed params loaded!")

        info_msg = f"Processing streams..."
        print(info_msg)
        self.logger.info(info_msg)

        if nav_data_estimator.config_parser.stream.record:
            fps = 1
            codec = cv2.VideoWriter_fourcc(*'FMP4')
            recording = cv2.VideoWriter(config_parser.stream.record_path, codec, fps, frame_size)

        compute_time = 0
        frame_count  = 0

        block_size = distance_calculator.config_parser.parameters.block_size
        max_disp   = distance_calculator.config_parser.parameters.max_disparities

        focal_length = distance_calculator.config_parser.left_camera_specs.focal_length
        pixel_size   = distance_calculator.config_parser.left_camera_specs.pixel_size
        baseline     = distance_calculator.config_parser.system_setup.baseline_distance

        stereo_sgbm = cv2.StereoSGBM_create(minDisparity=0, 
                                            numDisparities=max_disp,
                                            blockSize=block_size,
                                            P1=8 * 1 * block_size ** 2,
                                            P2=32 * 1 * block_size ** 2,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            disp12MaxDiff=1,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        stream_fps = min(stream_left.get_fps(), stream_right.get_fps())
        self.logger.info(f"Left stream running at {stream_left.get_fps()} fps.")
        self.logger.info(f"Right stream running at {stream_right.get_fps()} fps.")

        lost_frames = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while stream_left.isOpened() and stream_right.isOpened():
                detection_list = []
                
                ret_left, frame_left   = stream_left.read()
                ret_right, frame_right = stream_right.read()

                if not ret_right or not ret_left:
                    lost_frames += 1
                    if lost_frames >= config_parser.stream.lost_frames:
                        msg = f"Max lost frames reached ({config_parser.stream.lost_frames})."
                        self.logger.info(msg)
                        break
                    else:
                        time.sleep(1/stream_fps)
                        continue
                self.logger.info(f"Lost frames: {lost_frames}.")
                lost_frames = 0

                # Give Vessel detector some time to send latest data as we are reading latest
                # frame and detection emission takes some time to send
                #time.sleep(1)
                detection_buffer = nav_data_estimator.multicast_manager.detection_buffer

                if len(detection_buffer) == 0:
                    continue    # No detections to process
                
                while detection_buffer:
                    detection = detection_buffer.pop()
                    detection_list.append(detection)
                    self.logger.debug(f"Processing detection:")
                    self.logger.debug(f"    x: {detection.x}")
                    self.logger.debug(f"    y: {detection.y}")
                    self.logger.debug(f"    width: {detection.width}")
                    self.logger.debug(f"    height: {detection.height}")
                    self.logger.debug(f"    probability: {detection.probability}")
                
                correlation_tresh = config_parser.correlation.min_distance
                detection_list = self._correlate_detections(detection_list, correlation_tresh)

                remap_frame_start = time.time()
                frame_count += 1
            
                if not ret_left or not ret_right:
                    break

                computation_start_time = time.time()

                frame_right = cv2.resize(frame_right, None, fx=scale, fy=scale)
                frame_left  = cv2.resize(frame_left, None, fx=scale, fy=scale)

                frame_left_gray  = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

                rect_left  = cv2.remap(frame_left_gray, xmap_l, ymap_l, cv2.INTER_LINEAR)
                rect_right = cv2.remap(frame_right_gray, xmap_r, ymap_r, cv2.INTER_LINEAR)

                remap_frames_elapsed = time.time() - remap_frame_start
                msg = f"Time required to remap frames: {remap_frames_elapsed:.2f} secs."
                self.logger.debug(msg)
                
                stereo_compute_start = time.time()
                dispmap_comp = stereo_sgbm.compute(rect_left, rect_right)
                dispmap_sgbm = distance_calculator.apply_disparity_filter(dispmap_comp, stereo_sgbm,
                                                                            rect_left, rect_right)
                stereo_elapsed = time.time() - stereo_compute_start
                msg = f"Time required to compute stereo SGBM: {stereo_elapsed:.2f} secs."
                self.logger.debug(msg)
                
                homography_start = time.time()
                H = distance_calculator.get_homography(rect_left, frame_left)
                aligned_map = cv2.warpPerspective(dispmap_sgbm, H, (frame_left.shape[1], 
                                                                    frame_left.shape[0]))
                
                aligned_map = distance_calculator.get_distance_map(aligned_map, focal_length, 
                                                                   pixel_size, baseline)

                homography_elapsed = time.time() - homography_start
                msg = f"Time required to compute homography matrix: {homography_elapsed:.2f} secs."
                self.logger.debug(msg)
                
                distance_map = {}
                detection_id = 0
                for detection in detection_list:
                    detection_x = int(round(detection.x * frame_size[0]))
                    detection_y = int(round(detection.y * frame_size[1]))
                    detection_width  = int(round(detection.width * frame_size[0]))
                    detection_height = int(round(detection.height * frame_size[1]))
                    dist = float(distance_calculator.get_avg_distance(aligned_map, 
                                                                      (detection_y, detection_x),
                                                                      detection_width,
                                                                      detection_height))
                    self.logger.debug(f"Detection {(detection_x, detection_y)} distance: {dist}")
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"{current_time} - Detection {(detection.x, detection.y)} distance: {dist:.2f}")

                    if isnan(dist):
                        self.logger.debug(f"Skipping message")
                        continue

                    bearing, _ = distance_calculator.get_angle((detection_x, detection_y),
                                                               frame_size[0], frame_size[1])
                    
                    distance_map[(detection_x, detection_y)] = (dist, int(bearing), detection_width,
                                                                detection_height)
                    
                    nav_data_msg = NavData()
                    nav_data_msg.id = detection_id         # TODO Calculate
                    nav_data_msg.distance = dist / 100     # From cm to meters
                    nav_data_msg.bearing  = int(bearing)

                    self.logger.debug(f"NavData message to send:")
                    self.logger.debug(f"    id: {nav_data_msg.id}")
                    self.logger.debug(f"    distance: {nav_data_msg.distance}")
                    self.logger.debug(f"    bearing: {nav_data_msg.bearing}")

                    nav_data_estimator.multicast_manager.send_nav_data_async(nav_data_msg)
                    detection_id += 1

                computation_elapsed_time = time.time() - computation_start_time
                self.logger.debug(f"Computation time (loop): {computation_elapsed_time:.2f} secs.")

                compute_time += computation_elapsed_time
                compute_time_avg = compute_time / frame_count
                self.logger.debug(f"Computation time (avg): {compute_time_avg:.2f} secs.")

                future = executor.submit(self._get_recording_frame, distance_calculator,
                                             frame_left, aligned_map, distance_map, 
                                             config_parser.parameters.detection_kernel, frame_count)
                self.last_frame = future.result()

                if config_parser.stream.record:
                    recording.write(self.last_frame)   
            
            if config_parser.stream.show_video:
                self._stop_display_thread()
                cv2.destroyAllWindows()

            stream_left.release()
            stream_right.release()
            
            if config_parser.stream.record:
                recording.release()
            info_msg = f"Streamings are over!"
            print(info_msg)
            self.logger.info(info_msg)
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

    #log_filepath = f"./modules/NavDataEstimator/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = f"./prototype/20241005/logs/navdataestimator.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    #config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    config_filepath = f"./prototype/20241005/cfg/navdataestimator_cfg.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format, log_level=log_level, 
                  config_filepath=config_filepath, args=args)

    app.main()