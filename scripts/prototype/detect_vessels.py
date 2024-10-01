"""
TBD
"""

import argparse
import cv2
import os
import sys
import time

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.common.comms.detection import Detection, MAX_X, MAX_Y, MAX_WIDTH, MAX_HEIGHT, MAX_PROB
from modules.common.streamconsumer import StreamConsumer
from modules.VesselDetector.src.baseclass import BaseClass
from modules.VesselDetector.src.vesseldetector import VesselDetector
from modules.VesselDetector.src.utils.helpers import draw


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
            comms_out_log = file_name + "_comms_out" + file_extension
            with open(comms_out_log, 'w'):
                pass

        return res
    
    
    def _get_stream(self, config_parser):
        stream  = StreamConsumer(config_parser.stream.camera)

        # Wait until stream is available
        while not stream.isOpened() :
            stream.release()
            stream  = StreamConsumer(config_parser.stream.camera)
            time.sleep(1)

        return stream


    def _get_frame_size(self, stream):
        while stream.isOpened():
            # Obtain frame size by reading one frame from stream
            ret_left, frame_left = stream.read()
        
            if not ret_left:
                continue
            
            frame_width, frame_height = frame_left.shape[1], frame_left.shape[0]
            frame_size = (frame_width, frame_height)
            self.logger.info(f"Frame size: {frame_size}")
            return frame_size

        return None


    def main(self):
        info_msg = f"Running Vessel detector script..."
        print(info_msg)
        self.logger.info(info_msg)

        vessel_detector = VesselDetector(filename=self.log_filepath,  format=self.log_format,
                                         level=self.log_level, config_path=self.config_filepath)                         
        
        scale = vessel_detector.config_parser.stream.scale
        vessel_detector.multicast_manager.start_communications()
        
        info_msg = f"Waiting for camera stream..."
        print(info_msg)
        self.logger.info(info_msg)

        get_stream_start = time.time()
        stream = self._get_stream(vessel_detector.config_parser)
        get_stream_elapsed = time.time() - get_stream_start
        self.logger.info(f"Stream running at {stream.get_fps()} fps.") 
        self.logger.debug(f"Time required to connect to stream: {get_stream_elapsed:.2f} secs.")

        self.logger.info(f"Obtaining frame size...")

        frame_size = self._get_frame_size(stream)
        frame_size = (int(frame_size[0] * scale), int(frame_size[1] * scale))

        info_msg = f"Processing stream..."
        print(info_msg)
        self.logger.info(info_msg)

        compute_time = 0
        frame_count  = 0

        if vessel_detector.config_parser.stream.record:
            fps = 1
            codec = cv2.VideoWriter_fourcc(*'FMP4')
            recording = cv2.VideoWriter(vessel_detector.config_parser.stream.record_path, codec, 
                                        fps, frame_size)
                       
        lost_frames = 0
        while stream.isOpened():
            ret, frame = stream.read()
            frame_count += 1

            if not ret:
                lost_frames += 1
                if lost_frames >= vessel_detector.config_parser.stream.lost_frames:
                    max_lost = vessel_detector.config_parser.stream.lost_frames
                    msg = f"Max lost frames reached ({max_lost})."
                    self.logger.info(msg)
                    break
                else:
                    time.sleep(1/stream.get_fps())
                    continue

            self.logger.info(f"Lost frames: {lost_frames}.")
            lost_frames = 0
        
            computation_start_time = time.time()

            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            bbox_frame = frame.copy()

            detection_start = time.time()
            bboxes, class_names, confidences = vessel_detector.get_detections(frame)
            detection_elapsed = time.time() - detection_start
            self.logger.debug(f"Time required to detect vessels: {detection_elapsed:.2f} secs.")
            
            bboxes_abs = vessel_detector.get_bboxes_abs(img=frame, bboxes=bboxes)

            for i, bbox in enumerate(bboxes):
                confidence = confidences[i]
                class_name = class_names[i]
                bbox_abs   = bboxes_abs[i]

                self.logger.debug(f"Detection bbox: {bbox}")
                self.logger.debug(f"Detection class_name: {class_name}")
                self.logger.debug(f"Detection confidence: {confidence}")

                allowed_class_names = vessel_detector.config_parser.detection.detection_names
                if class_name not in allowed_class_names and 'all' not in allowed_class_names:
                    msg = f"Detected object ({class_name}) is not a vessel " +\
                          f"({allowed_class_names}), discarded."
                    self.logger.debug(msg)
                    continue

                x = bbox[0]
                y = bbox[1]
                width  = bbox[2]
                height = bbox[3]

                if x > MAX_X:
                    msg = f"Detection x ({x}) higher than limit ({MAX_X}) setting to {MAX_X}."
                    self.logger.warning(msg)
                    x = MAX_X
                
                if y > MAX_Y:
                    msg = f"Detection y ({y}) higher than limit ({MAX_Y}) setting to {MAX_Y}."
                    self.logger.warning(msg)
                    y = MAX_Y
                
                if width > MAX_WIDTH:
                    msg = f"Detection width ({width}) higher than limit ({MAX_WIDTH}) setting " +\
                        f"to {MAX_WIDTH}."
                    self.logger.warning(msg)
                    width = MAX_WIDTH
                
                if height > MAX_HEIGHT:
                    msg = f"Detection height ({height}) higher than limit ({MAX_HEIGHT})  " +\
                        f"setting to {MAX_HEIGHT}."
                    self.logger.warning(msg)
                    height = MAX_HEIGHT
                
                if confidence > MAX_PROB:
                    msg = f"Detection probability ({confidence}) higher than limit ({MAX_PROB}) " +\
                        f"setting to {MAX_PROB}."
                    self.logger.warning(msg)
                    confidence = MAX_PROB

                if confidence > vessel_detector.config_parser.detection.min_confidence:
                    detection_msg = Detection()
                    detection_msg.x = x
                    detection_msg.y = y
                    detection_msg.width  = width
                    detection_msg.height = height
                    detection_msg.probability = confidence

                    self.logger.debug(f"Detection message to send:")
                    self.logger.debug(f"    x: {detection_msg.x}")
                    self.logger.debug(f"    y: {detection_msg.y}")
                    self.logger.debug(f"    width: {detection_msg.width}")
                    self.logger.debug(f"    height: {detection_msg.height}")
                    self.logger.debug(f"    probability: {detection_msg.probability}")
                    
                    vessel_detector.multicast_manager.send_detection(detection_msg)
                    if vessel_detector.config_parser.stream.record:
                        bbox_frame = draw(bbox_abs, bbox_frame)

                        ##################### DEBUG DELETE #####################
                        from datetime import datetime
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        cv2.putText(bbox_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    (0, 0, 255), 2, cv2.LINE_AA) 
                        cv2.putText(bbox_frame, f"Time: {current_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255), 2, cv2.LINE_AA)
                         
                        ########################################################

                        recording.write(bbox_frame)

                computation_elapsed_time = time.time() - computation_start_time
                self.logger.debug(f"Computation time (loop): {computation_elapsed_time:.2f} secs.")
                compute_time += computation_elapsed_time
                
            compute_time_avg = compute_time / frame_count
            self.logger.debug(f"Computation time (avg): {compute_time_avg:.2f} secs.")
        
        if vessel_detector.config_parser.stream.record:
            recording.release()

        stream.release()

        info_msg = f"Streaming is over!"
        print(info_msg)
        self.logger.info(info_msg)


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

    log_filepath = f"./modules/VesselDetector/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format   = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level    = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/VesselDetector/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format, log_level=log_level, 
                  config_filepath=config_filepath, args=args)

    app.main()