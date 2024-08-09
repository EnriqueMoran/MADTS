"""
TBD
"""

import configparser
import logging

from pathlib import Path


__author__ = "EnriqueMoran"


logger = logging.getLogger("ConfigManager")


class ConfigManager():
    """
    TBD
    """

    def __init__(self, config_path="./cfg/config.ini"):
        self.config_path = Path(config_path).resolve()

        ##### RTMP SERVER #####
        self.gopro_right_in_url = None
        self.gopro_left_in_url  = None
        

        ##### STREAMING  #####
        self.gopro_right_out_url = None
        self.gopro_left_out_url  = None
        

        ##### GOPRO MANAGEMENT #####
        self.gopro_right_name = None
        self.gopro_left_name = None
        
        
    def read_config(self) -> bool:
        """
        TBD
        """
        logger.info(f"Reading configuration from {self.config_path}")

        if not self.config_path.exists():
            logger.warning(f"Configuration file {self.config_path} does not exist.")
            return False

        config = configparser.ConfigParser(inline_comment_prefixes=";")
        config.read(self.config_path)

        res = True    # All configuration could be read succesfully

        try:
            self.gopro_left_in_url = str(config.get("RTMP_SERVER", "gopro_left_url"))
            logger.info(f"Read RTMP_SERVER - gopro_left_url: {self.gopro_left_in_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_left_url' in section 'RTMP_SERVER': {e}")
            res = False

        try:
            self.gopro_right_in_url = str(config.get("RTMP_SERVER", "gopro_right_url"))
            logger.info(f"Read RTMP_SERVER - gopro_right_url: {self.gopro_right_in_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_right_url' in section 'RTMP_SERVER': {e}")
            res = False

        try:
            self.gopro_left_out_url = str(config.get("STREAMING", "gopro_left_url"))
            logger.info(f"Read STREAMING - gopro_left_url: {self.gopro_left_out_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_left_url' in section 'STREAMING': {e}")
            res = False

        try:
            self.gopro_right_out_url = str(config.get("STREAMING", "gopro_right_url"))
            logger.info(f"Read STREAMING - gopro_right_url: {self.gopro_right_out_url}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_right_url' in section 'STREAMING': {e}")
            res = False

        try:
            self.gopro_left_name = str(config.get("GOPRO_MANAGEMENT", "gopro_left_name"))
            logger.info(f"Read GOPRO_MANAGEMENT - gopro_left_name: {self.gopro_left_name}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_left_name' in section 'GOPRO_MANAGEMENT': {e}")
            res = False
        
        try:
            self.gopro_right_name = str(config.get("GOPRO_MANAGEMENT", "gopro_right_name"))
            logger.info(f"Read GOPRO_MANAGEMENT - gopro_right_name: {self.gopro_right_name}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.warning(f"Could not find 'gopro_right_name' in section 'GOPRO_MANAGEMENT': {e}")
            res = False 

        logger.info(f"Finished reading configuration.")
        return res

        