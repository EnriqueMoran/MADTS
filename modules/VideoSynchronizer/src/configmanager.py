"""
Implements configuration parser class.
"""

import configparser
import src.utils.globals

from pathlib import Path

from src.baseclass import BaseClass


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
            self.gopro_right_in_url = None
            self.gopro_left_in_url  = None
            self.network_ssid       = None
            self.network_password   = None
            
            ##### GOPRO MANAGEMENT #####
            self.gopro_right_name = None
            self.gopro_left_name  = None
            self.record_stream    = None

            ##### STREAM MANAGEMENT #####
            self.resolution  = None
            self.min_bitrate = None
            self.max_bitrate = None
            self.starting_bitrate = None
            self.fov = None
            self.fps = None            # TODO
            self.duration = None       # TODO
            self.hypersmooth = None    # TODO

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
            self.gopro_left_in_url = str(config.get("RTMP_SERVER", "gopro_left_url"))
            self.logger.info(f"Read RTMP_SERVER - gopro_left_url: {self.gopro_left_in_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_left_url' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.gopro_right_in_url = str(config.get("RTMP_SERVER", "gopro_right_url"))
            self.logger.info(f"Read RTMP_SERVER - gopro_right_url: {self.gopro_right_in_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_right_url' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.network_ssid = str(config.get("RTMP_SERVER", "network_ssid"))
            self.logger.info(f"Read RTMP_SERVER - network_ssid: {self.network_ssid}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'network_ssid' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.network_password = str(config.get("RTMP_SERVER", "network_password"))
            self.logger.info(f"Read RTMP_SERVER - network_password: {self.network_password}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'network_password' in section 'RTMP_SERVER': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.gopro_left_name = str(config.get("GOPRO_MANAGEMENT", "gopro_left_name"))
            self.logger.info(f"Read GOPRO_MANAGEMENT - gopro_left_name: {self.gopro_left_name}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_left_name' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.gopro_right_name = str(config.get("GOPRO_MANAGEMENT", "gopro_right_name"))
            self.logger.info(f"Read GOPRO_MANAGEMENT - gopro_right_name: {self.gopro_right_name}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_right_name' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False 
        
        try:
            self.record_stream = bool(int(config.get("GOPRO_MANAGEMENT", "record_stream")))
            self.logger.info(f"Read GOPRO_MANAGEMENT - record_stream: {self.record_stream}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'record_stream' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.resolution = int(config.get("STREAM", "resolution"))
            self.logger.info(f"Read STREAM - resolution: {self.resolution}.")

            if self.resolution not in src.utils.globals.RESOLUTION:
                warning_msg = f"Value not admitted, valid values: {src.utils.globals.RESOLUTION},"+\
                              f" setting value to {src.utils.globals.RESOLUTION[0]}."
                self.logger.warning(warning_msg)
                self.resolution = src.utils.globals.RESOLUTION
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'resolution' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.min_bitrate = int(config.get("STREAM", "min_bitrate"))
            self.logger.info(f"Read STREAM - min_bitrate: {self.min_bitrate}.")

            if self.min_bitrate < src.utils.globals.MIN_BITRATE:
                warning_msg = f"Minimum valid value: {src.utils.globals.MIN_BITRATE}, setting value " +\
                              f"to {src.utils.globals.MIN_BITRATE}."
                self.logger.warning(warning_msg)
                self.min_bitrate = src.utils.globals.MIN_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'min_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.max_bitrate = int(config.get("STREAM", "max_bitrate"))
            self.logger.info(f"Read STREAM - max_bitrate: {self.max_bitrate}.")

            if self.max_bitrate > src.utils.globals.MAX_BITRATE:
                warning_msg = f"Maximum valid value: {src.utils.globals.MAX_BITRATE}, setting value " +\
                              f"to {src.utils.globals.MAX_BITRATE}."
                self.logger.warning(warning_msg)
                self.max_bitrate = src.utils.globals.MAX_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'max_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.starting_bitrate = int(config.get("STREAM", "starting_bitrate"))
            self.logger.info(f"Read STREAM - starting_bitrate: {self.starting_bitrate}.")

            if self.starting_bitrate < src.utils.globals.MIN_BITRATE:
                warning_msg = f"Minimum valid value: {src.utils.globals.MIN_BITRATE}, setting value " +\
                              f"to {src.utils.globals.MIN_BITRATE}."
                self.logger.warning(warning_msg)
                self.starting_bitrate = src.utils.globals.MIN_BITRATE
            
            if self.starting_bitrate > src.utils.globals.MAX_BITRATE:
                warning_msg = f"Maximum valid value: {src.utils.globals.MAX_BITRATE}, setting value " +\
                              f"to {src.utils.globals.MAX_BITRATE}."
                self.logger.warning(warning_msg)
                self.starting_bitrate = src.utils.globals.MAX_BITRATE
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'starting_bitrate' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.fov = int(config.get("STREAM", "fov"))
            self.logger.info(f"Read STREAM - fov: {self.fov}.")

            if self.fov not in src.utils.globals.FOV:
                warning_msg = f"Value not admitted, valid values: {src.utils.globals.FOV}, "+\
                    f"setting value to {src.utils.globals.FOV[0]}."
                self.logger.warning(warning_msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'fov' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res

        