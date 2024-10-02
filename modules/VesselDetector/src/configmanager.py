"""
Implements configuration parser class.
"""

import configparser

from pathlib import Path

from modules.VesselDetector.src.baseclass import BaseClass
from modules.VesselDetector.src.utils.structs import *


__author__ = "EnriqueMoran"


class ConfigManager(BaseClass):
    """
    This class reads from configuration file and parse its values.

    Args:
        - config_path (str): Path to configuration file.
        - filename (str): Path to store log file; belongs to BaseClass.
        - format (str): Logger format; belongs to BaseClass.
        - level (str): Logger level; belongs to BaseClass.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, filename:str, format:str, level:str, config_path:str):
        if not hasattr(self, 'initialized'):
            super().__init__(filename, format, level)
            self.config_path = Path(config_path).resolve()

            ##### MODEL #####
            self.model = Model()

            #### COMMUNICATION OUT ####
            self.comm_out = Communication()

            #### STREAM ####
            self.stream = Stream()

            #### DETECTION ####
            self.detection = Detection()

            self.read_config()

            self.initialized = True


    def read_config(self) -> bool:
        """
        Read configuration file and returns wether there was any error while doing it.

        Returns:
            - res (bool): True if all parameters could be read successfully, false otherwise.
        """
        self.logger.info(f"Reading configuration from {self.config_path}")

        if not self.config_path.exists():
            self.logger.warning(f"Configuration file {self.config_path} does not exist.")
            return False

        config = configparser.ConfigParser(inline_comment_prefixes=";")
        config.read(self.config_path)

        res = True    # All configuration could be read succesfully
        
        try:
            self.model.model_config_file = str(config.get(
                                                   "MODEL",
                                                   "model_config_file").strip()
                                              )
            msg = f"Read MODEL - model_config_file: {self.model.model_config_file}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'model_config_file' in " +\
                          f"section 'MODEL': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.model.weights_file = str(config.get(
                                              "MODEL",
                                              "weights_file").strip()
                                         )
            msg = f"Read MODEL - weights_file: {self.model.weights_file}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'weights_file' in " +\
                          f"section 'MODEL': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.model.class_names_file = str(config.get(
                                                  "MODEL",
                                                  "class_names_file").strip()
                                             )
            msg = f"Read MODEL - class_names_file: {self.model.class_names_file}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'class_names_file' in " +\
                          f"section 'MODEL': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.model.width = int(config.get(
                                             "MODEL",
                                             "image_width").strip()
                                        )
            msg = f"Read MODEL - image_width: {self.model.width}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_width' in " +\
                          f"section 'MODEL': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.model.height = int(config.get(
                                              "MODEL",
                                              "image_height").strip()
                                         )
            msg = f"Read MODEL - image_height: {self.model.height}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_height' in " +\
                          f"section 'MODEL': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.comm_out.group = str(config.get(
                                          "COMMUNICATION_OUT",
                                          "multicast_group").strip()
                                     )

            msg = f"Read COMMUNICATION_OUT - multicast_group: {self.comm_out.group}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_group' in section 'COMMUNICATION_OUT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.comm_out.port = int(config.get(
                                         "COMMUNICATION_OUT", 
                                         "multicast_port").strip()
                                    )

            msg = f"Read COMMUNICATION_OUT - multicast_port: {self.comm_out.port}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_port' in section 'COMMUNICATION_OUT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.comm_out.iface = str(config.get(
                                          "COMMUNICATION_OUT", 
                                          "multicast_iface").strip()
                                     )
                                                                     
            msg = f"Read COMMUNICATION_OUT - multicast_iface: {self.comm_out.iface}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_iface' in section 'COMMUNICATION_OUT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.comm_out.ttl = int(config.get(
                                        "COMMUNICATION_OUT", 
                                        "multicast_ttl").strip()
                                    )

            msg = f"Read COMMUNICATION_OUT - multicast_ttl: {self.comm_out.ttl}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_ttl' in section 'COMMUNICATION_OUT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.camera = str(config.get(
                                              "STREAM", 
                                              "camera_url").strip()
                                         )
                                                                     
            msg = f"Read STREAM - camera_url: {self.stream.camera}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'camera_url' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.scale = float(config.get(
                                          "STREAM", 
                                          "scale").strip()
                                     )   
                                                                     
            msg = f"Read STREAM - scale: {self.stream.scale}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'scale' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.record = bool(int(config.get(
                                              "STREAM", 
                                              "record_stream").strip())
                                     )
            msg = f"Read STREAM - record_stream: {self.stream.record}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'record_stream' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.video_width = int(config.get(
                                              "STREAM", 
                                              "video_width").strip()
                                         )

            msg = f"Read STREAM - video_width: {self.stream.video_width}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'video_width' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.video_height = int(config.get(
                                               "STREAM", 
                                               "video_height").strip()
                                          )

            msg = f"Read STREAM - video_height: {self.stream.video_height}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'video_height' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.show_video = bool(int(config.get(
                                              "STREAM", 
                                              "show_video").strip())
                                     )
            msg = f"Read STREAM - show_video: {self.stream.show_video}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'show_video' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.record_path = str(config.get(
                                              "STREAM", 
                                              "record_filepath").strip()
                                         )
            msg = f"Read STREAM - record_filepath: {self.stream.record_path}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'record_filepath' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.lost_frames = int(config.get(
                                               "STREAM", 
                                               "max_lost_frames").strip()
                                          )   
                                                                     
            msg = f"Read STREAM - max_lost_frames: {self.stream.lost_frames}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'max_lost_frames' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.detection.min_confidence = float(config.get(
                                                    "DETECTION", 
                                                    "min_confidence").strip()
                                               )   
                                                                     
            msg = f"Read DETECTION - min_confidence: {self.detection.min_confidence}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'min_confidence' in section 'DETECTION': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            value = str(config.get(
                            "DETECTION", 
                            "class_names").strip()
                       )
            msg = f"Read DETECTION - class_names: {value}"
            self.logger.info(msg)
            self.detection.detection_names = [item.strip().lower() for item in value.split(',')]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'class_names' in section 'DETECTION': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}")
        return res
        
        