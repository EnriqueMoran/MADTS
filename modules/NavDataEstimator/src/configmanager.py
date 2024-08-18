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

            ##### SYSTEM_SETUP #####
            self.baseline_distance = None

            ##### CALIBRATION_LEFT #####
            self.left_image_directory   = None
            self.left_chessboard_width  = None
            self.left_chessboard_height = None
            self.left_square_size       = None
            self.left_frame_width       = None
            self.left_frame_height      = None
            self.left_save_calibrated   = None
            self.left_params_directory  = None
            
            ##### CALIBRATION_RIGHT #####
            self.right_image_directory   = None
            self.right_chessboard_width  = None
            self.right_chessboard_height = None
            self.right_square_size       = None
            self.right_frame_width       = None
            self.right_frame_height      = None
            self.right_save_calibrated   = None
            self.right_params_directory  = None

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
            self.baseline_distance = float(config.get("SYSTEM_SETUP", "baseline_distance"))
            self.logger.info(f"Read SYSTEM_SETUP - baseline_distance: {self.fov}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'baseline_distance' in section 'SYSTEM_SETUP': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_image_directory = str(config.get("CALIBRATION_LEFT", "image_directory"))
            self.logger.info(f"Read CALIBRATION_LEFT - image_directory: {self.left_image_directory}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_directory' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_chessboard_width = int(config.get("CALIBRATION_LEFT", "chessboard_width"))
            self.logger.info(f"Read CALIBRATION_LEFT - chessboard_width: {self.left_chessboard_width}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_width' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_chessboard_height = int(config.get("CALIBRATION_LEFT", "chessboard_height"))
            self.logger.info(f"Read CALIBRATION_LEFT - chessboard_height: {self.left_chessboard_height}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_height' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_square_size = int(config.get("CALIBRATION_LEFT", "chessboard_square_size"))
            self.logger.info(f"Read CALIBRATION_LEFT - chessboard_square_size: {self.left_square_size}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_square_size' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_frame_width = int(config.get("CALIBRATION_LEFT", "frame_width"))
            self.logger.info(f"Read CALIBRATION_LEFT - frame_width: {self.left_frame_width}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_width' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_frame_height = int(config.get("CALIBRATION_LEFT", "frame_height"))
            self.logger.info(f"Read CALIBRATION_LEFT - frame_height: {self.left_frame_height}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_height' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_save_calibrated = bool(int(config.get("CALIBRATION_LEFT", "save_calibrated_image")))
            self.logger.info(f"Read CALIBRATION_LEFT - save_calibrated_image: {self.left_save_calibrated}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibrated_image' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_params_directory = str(config.get("CALIBRATION_LEFT", "save_calibration_params_path"))
            self.logger.info(f"Read CALIBRATION_LEFT - save_calibration_params_path: {self.left_params_directory}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params_path' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_image_directory = str(config.get("CALIBRATION_RIGHT", "image_directory"))
            self.logger.info(f"Read CALIBRATION_RIGHT - image_directory: {self.right_image_directory}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_directory' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_chessboard_width = int(config.get("CALIBRATION_RIGHT", "chessboard_width"))
            self.logger.info(f"Read CALIBRATION_RIGHT - chessboard_width: {self.right_chessboard_width}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_width' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_chessboard_height = int(config.get("CALIBRATION_RIGHT", "chessboard_height"))
            self.logger.info(f"Read CALIBRATION_RIGHT - chessboard_height: {self.right_chessboard_height}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_height' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_square_size = int(config.get("CALIBRATION_RIGHT", "chessboard_square_size"))
            self.logger.info(f"Read CALIBRATION_RIGHT - chessboard_square_size: {self.right_square_size}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_square_size' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_frame_width = int(config.get("CALIBRATION_RIGHT", "frame_width"))
            self.logger.info(f"Read CALIBRATION_RIGHT - frame_width: {self.right_frame_width}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_width' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_frame_height = int(config.get("CALIBRATION_RIGHT", "frame_height"))
            self.logger.info(f"Read CALIBRATION_RIGHT - frame_height: {self.right_frame_height}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_height' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_save_calibrated = bool(int(config.get("CALIBRATION_RIGHT", "save_calibrated_image")))
            self.logger.info(f"Read CALIBRATION_RIGHT - save_calibrated_image: {self.right_save_calibrated}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibrated_image' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_params_directory = str(config.get("CALIBRATION_RIGHT", "save_calibration_params_path"))
            self.logger.info(f"Read CALIBRATION_RIGHT - save_calibration_params_path: {self.right_params_directory}.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params_path' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res

        