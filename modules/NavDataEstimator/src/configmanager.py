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

            ##### LEFT_CAMERA #####
            self.left_focal_lenght = None
            self.left_pixel_size   = None
            self.left_fov          = None
            
            ##### RIGHT_CAMERA #####
            self.right_focal_lenght = None
            self.right_pixel_size   = None
            self.right_fov          = None

            ##### CALIBRATION #####
            self.baseline_distance = None
            
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
            self.left_focal_lenght = int(config.get("LEFT_CAMERA", "focal_length"))
            self.logger.info(f"Read LEFT_CAMERA - focal_length: {self.left_focal_lenght}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_pixel_size = float(config.get("LEFT_CAMERA", "pixel_size"))
            self.logger.info(f"Read LEFT_CAMERA - pixel_size: {self.left_pixel_size}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.fov = float(config.get("LEFT_CAMERA", "horizontal_fov"))
            self.logger.info(f"Read LEFT_CAMERA - horizontal_fov: {self.fov}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_focal_lenght = int(config.get("RIGHT_CAMERA", "focal_length"))
            self.logger.info(f"Read RIGHT_CAMERA - focal_length: {self.right_focal_lenght}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_pixel_size = float(config.get("RIGHT_CAMERA", "pixel_size"))
            self.logger.info(f"Read RIGHT_CAMERA - pixel_size: {self.right_pixel_size}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.fov = float(config.get("RIGHT_CAMERA", "horizontal_fov"))
            self.logger.info(f"Read RIGHT_CAMERA - horizontal_fov: {self.fov}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.baseline_distance = float(config.get("CALIBRATION", "baseline_distance"))
            self.logger.info(f"Read CALIBRATION - baseline_distance: {self.fov}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'baseline_distance' in section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res

        