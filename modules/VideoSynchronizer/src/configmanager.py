"""
Implements configuration parser class.
"""

import configparser

from pathlib import Path

from modules.VideoSynchronizer.src.baseclass import BaseClass
from modules.VideoSynchronizer.src.utils.globals import *
from modules.VideoSynchronizer.src.utils.structs import *


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

            ##### RTMP SERVER #####
            self.rtmp = RTMP()
            
            ##### GOPRO MANAGEMENT #####
            self.gopro = GoPro()

            ##### STREAM MANAGEMENT #####
            self.stream = Stream()

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
            self.rtmp.left_camera_url = str(config.get(
                                                "RTMP_SERVER", 
                                                "gopro_left_url").strip()
                                            )
            self.logger.info(f"Read RTMP_SERVER - gopro_left_url: {self.rtmp.left_camera_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_left_url' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.rtmp.right_camera_url = str(config.get(
                                                 "RTMP_SERVER", 
                                                 "gopro_right_url").strip()
                                            )
            self.logger.info(f"Read RTMP_SERVER - gopro_right_url: {self.rtmp.right_camera_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_right_url' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.rtmp.network_ssid = str(config.get(
                                             "RTMP_SERVER", 
                                             "network_ssid").strip()
                                        )
            self.logger.info(f"Read RTMP_SERVER - network_ssid: {self.rtmp.network_ssid}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'network_ssid' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.rtmp.network_password = str(config.get(
                                                 "RTMP_SERVER", 
                                                 "network_password").strip()
                                            )
            self.logger.info(f"Read RTMP_SERVER - network_password: {self.rtmp.network_password}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'network_password' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.gopro.left_camera_name = str(config.get(
                                                  "GOPRO_MANAGEMENT", 
                                                  "gopro_left_name").strip()
                                             )
            msg = f"Read GOPRO_MANAGEMENT - gopro_left_name: {self.gopro.left_camera_name}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_left_name' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.gopro.right_camera_name = str(config.get(
                                                   "GOPRO_MANAGEMENT", 
                                                   "gopro_right_name").strip()
                                              )
            msg = f"Read GOPRO_MANAGEMENT - gopro_right_name: {self.gopro.right_camera_name}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_right_name' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False 

        try:
            self.stream.resolution = int(config.get(
                                             "STREAM", 
                                             "resolution").strip()
                                        )
            self.logger.info(f"Read STREAM - resolution: {self.stream.resolution}.")

            if self.stream.resolution not in RESOLUTION:
                warning_msg = f"Value not admitted, valid values: {RESOLUTION},"+\
                              f" setting value to {RESOLUTION[0]}."
                self.logger.warning(warning_msg)
                self.stream.resolution = RESOLUTION
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'resolution' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.min_bitrate = int(config.get(
                                              "STREAM", 
                                              "min_bitrate").strip()
                                         )
            self.logger.info(f"Read STREAM - min_bitrate: {self.stream.min_bitrate}.")

            if self.stream.min_bitrate < MIN_BITRATE:
                warning_msg = f"Minimum valid value: {MIN_BITRATE}, " +\
                              f"setting value to {MIN_BITRATE}."
                self.logger.warning(warning_msg)
                self.stream.min_bitrate = MIN_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'min_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.max_bitrate = int(config.get(
                                              "STREAM", 
                                              "max_bitrate").strip()
                                         )
            self.logger.info(f"Read STREAM - max_bitrate: {self.stream.max_bitrate}.")

            if self.stream.max_bitrate > MAX_BITRATE:
                warning_msg = f"Maximum valid value: {MAX_BITRATE}, " +\
                              f"setting value to {MAX_BITRATE}."
                self.logger.warning(warning_msg)
                self.stream.max_bitrate = MAX_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'max_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.starting_bitrate = int(config.get(
                                                   "STREAM", 
                                                   "starting_bitrate").strip()
                                              )
            self.logger.info(f"Read STREAM - starting_bitrate: {self.stream.starting_bitrate}.")

            if self.stream.starting_bitrate < MIN_BITRATE:
                warning_msg = f"Minimum valid value: {MIN_BITRATE}, " +\
                              f"setting value to {MIN_BITRATE}."
                self.logger.warning(warning_msg)
                self.stream.starting_bitrate = MIN_BITRATE
            
            if self.stream.starting_bitrate > MAX_BITRATE:
                warning_msg = f"Maximum valid value: {MAX_BITRATE}, " +\
                              f"setting value to {MAX_BITRATE}."
                self.logger.warning(warning_msg)
                self.stream.starting_bitrate = MAX_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'starting_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.fov = int(config.get(
                                      "STREAM", 
                                      "fov").strip()
                                  )
            self.logger.info(f"Read STREAM - fov: {self.stream.fov}.")

            if self.stream.fov not in FOV:
                warning_msg = f"Value not admitted, valid values: {FOV}, "+\
                    f"setting value to {FOV[0]}."
                self.logger.warning(warning_msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'fov' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.fps = int(config.get(
                                      "STREAM", 
                                      "fps").strip()
                                 )
            self.logger.info(f"Read STREAM - fps: {self.stream.fps}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'fps' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.duration = int(config.get(
                                           "STREAM", 
                                           "duration").strip()
                                      )
            self.logger.info(f"Read STREAM - duration: {self.stream.duration}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'duration' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.record_stream = bool(int(config.get(
                                                     "STREAM", 
                                                     "record_stream").strip())
                                            )
            self.logger.info(f"Read STREAM - record_stream: {self.stream.record_stream}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'record_stream' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res