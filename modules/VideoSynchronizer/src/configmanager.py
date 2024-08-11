"""
Implements configuration parser class.
"""

import configparser

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
            
            ##### STREAMING  #####
            self.gopro_right_out_url = None
            self.gopro_left_out_url  = None
            
            ##### GOPRO MANAGEMENT #####
            self.gopro_right_name = None
            self.gopro_left_name  = None
            self.record_stream    = None
            self.use_http         = None

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
            self.gopro_left_out_url = str(config.get("STREAMING", "gopro_left_url"))
            self.logger.info(f"Read STREAMING - gopro_left_url: {self.gopro_left_out_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_left_url' in section 'STREAMING': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.gopro_right_out_url = str(config.get("STREAMING", "gopro_right_url"))
            self.logger.info(f"Read STREAMING - gopro_right_url: {self.gopro_right_out_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gopro_right_url' in section 'STREAMING': {e}"
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
            self.use_http = bool(int(config.get("GOPRO_MANAGEMENT", "use_http")))
            self.logger.info(f"Read GOPRO_MANAGEMENT - use_http: {self.use_http}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'use_http' in section 'GOPRO_MANAGEMENT': {e}"
            self.logger.warning(warning_msg)
            res = False 

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res

        