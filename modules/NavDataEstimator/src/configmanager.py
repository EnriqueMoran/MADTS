"""
Implements configuration parser class.
"""

import configparser

from pathlib import Path
from src.baseclass import BaseClass
from src.utils.structs import *


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
            self.left_camera_specs = CameraSpecs(None, None, None)
            
            ##### RIGHT_CAMERA #####
            self.right_camera_specs = CameraSpecs(None, None, None)

            ##### SYSTEM_SETUP #####
            self.system_setup = SystemSetup(None)

            ##### CALIBRATION_LEFT #####
            self.left_camera_calibration = CalibrationSpecs(None, None, None, None, 
                                                            None, None, None, None)
            
            ##### CALIBRATION_RIGHT #####
            self.right_camera_calibration = CalibrationSpecs(None, None, None, None, 
                                                             None, None, None, None)
            
            ##### PARAMETERS #####
            self.parameters = Parameters(None, None, None, None)

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
            self.left_camera_specs.focal_length = float(config.get("LEFT_CAMERA_SPECS", 
                                                                   "focal_length"))
            msg = f"Read LEFT_CAMERA - focal_length: {self.left_camera_specs.focal_length}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_camera_specs.pixel_size = float(config.get("LEFT_CAMERA_SPECS", "pixel_size"))
            msg = f"Read LEFT_CAMERA - pixel_size: {self.left_camera_specs.pixel_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_camera_specs.fov = float(config.get("LEFT_CAMERA_SPECS", "horizontal_fov"))
            msg = f"Read LEFT_CAMERA - horizontal_fov: {self.left_camera_specs.fov}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_specs.focal_length = int(config.get("RIGHT_CAMERA_SPECS", 
                                                                 "focal_length"))
            msg = f"Read RIGHT_CAMERA - focal_length: {self.right_camera_specs.focal_length}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_camera_specs.pixel_size = float(config.get("RIGHT_CAMERA_SPECS", 
                                                                  "pixel_size"))
            msg = f"Read RIGHT_CAMERA - pixel_size: {self.right_camera_specs.pixel_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_camera_specs.fov = float(config.get("RIGHT_CAMERA_SPECS", "horizontal_fov"))
            msg = f"Read RIGHT_CAMERA - horizontal_fov: {self.right_camera_specs.fov}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.system_setup.baseline_distance = float(config.get("SYSTEM_SETUP", 
                                                                   "baseline_distance"))
            msg = f"Read SYSTEM_SETUP - baseline_distance: {self.system_setup.baseline_distance}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'baseline_distance' in section 'SYSTEM_SETUP': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.image_directory = str(config.get("CALIBRATION_LEFT", 
                                                                          "image_directory"))
            msg = f"Read CALIBRATION_LEFT - image_directory: " +\
                  f"{self.left_camera_calibration.image_directory}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_directory' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.chessboard_width = int(config.get("CALIBRATION_LEFT", 
                                                                           "chessboard_width"))
            msg = f"Read CALIBRATION_LEFT - chessboard_width: " +\
                  f"{self.left_camera_calibration.chessboard_width}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_width' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.chessboard_height = int(config.get("CALIBRATION_LEFT", 
                                                                            "chessboard_height"))
            msg = f"Read CALIBRATION_LEFT - chessboard_height: " +\
                  f"{self.left_camera_calibration.chessboard_height}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_height' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.chessboard_square_size = int(config.get(
                                                                        "CALIBRATION_LEFT",
                                                                        "chessboard_square_size")
                                                                     )
            msg = f"Read CALIBRATION_LEFT - chessboard_square_size: " +\
                  f"{self.left_camera_calibration.chessboard_square_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_square_size' in " +\
                          f"section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.frame_width = int(config.get("CALIBRATION_LEFT", 
                                                                      "frame_width"))
            msg = f"Read CALIBRATION_LEFT - frame_width: " +\
                  f"{self.left_camera_calibration.frame_width}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_width' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.frame_height = int(config.get("CALIBRATION_LEFT", 
                                                                       "frame_height"))
            msg = f"Read CALIBRATION_LEFT - frame_height: " +\
                  f"{self.left_camera_calibration.frame_height}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_height' in section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_camera_calibration.save_calibrated_image = bool(int(config.get(
                                                                        "CALIBRATION_LEFT", 
                                                                        "save_calibrated_image"))
                                                                     )
            msg = f"Read CALIBRATION_LEFT - save_calibrated_image: " +\
                  f"{self.left_camera_calibration.save_calibrated_image}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibrated_image' in " +\
                          f"section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.left_camera_calibration.save_calibration_params_path = str(config.get(
                                                                    "CALIBRATION_LEFT", 
                                                                    "save_calibration_params_path")
                                                                           )
            msg = f"Read CALIBRATION_LEFT - save_calibration_params_path: " +\
                  f"{self.left_camera_calibration.save_calibration_params_path}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params_path' in " +\
                          f"section 'CALIBRATION_LEFT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.image_directory = str(config.get("CALIBRATION_RIGHT", 
                                                                          "image_directory"))
            msg = f"Read CALIBRATION_RIGHT - image_directory: " +\
                  f"{self.right_camera_calibration.image_directory}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'image_directory' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.chessboard_width = int(config.get("CALIBRATION_RIGHT", 
                                                                           "chessboard_width"))
            msg = f"Read CALIBRATION_RIGHT - chessboard_width: " +\
                  f"{self.right_camera_calibration.chessboard_width}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_width' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.chessboard_height = int(config.get("CALIBRATION_RIGHT", 
                                                                            "chessboard_height"))
            msg = f"Read CALIBRATION_RIGHT - chessboard_height: " +\
                  f"{self.right_camera_calibration.chessboard_height}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_height' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.chessboard_square_size = int(config.get(
                                                                        "CALIBRATION_RIGHT",
                                                                        "chessboard_square_size")
                                                                     )
            msg = f"Read CALIBRATION_RIGHT - chessboard_square_size: " +\
                  f"{self.right_camera_calibration.chessboard_square_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_square_size' in " +\
                          f"section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.frame_width = int(config.get("CALIBRATION_RIGHT", 
                                                                      "frame_width"))
            msg = f"Read CALIBRATION_RIGHT - frame_width: " +\
                  f"{self.right_camera_calibration.frame_width}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_width' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.frame_height = int(config.get("CALIBRATION_RIGHT", 
                                                                       "frame_height"))
            msg = f"Read CALIBRATION_RIGHT - frame_height: " +\
                  f"{self.right_camera_calibration.frame_height}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'frame_height' in section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_camera_calibration.save_calibrated_image = bool(int(config.get(
                                                                        "CALIBRATION_RIGHT", 
                                                                        "save_calibrated_image"))
                                                                     )
            msg = f"Read CALIBRATION_RIGHT - save_calibrated_image: " +\
                  f"{self.right_camera_calibration.save_calibrated_image}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibrated_image' in " +\
                          f"section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_calibration.save_calibration_params_path = str(config.get(
                                                                    "CALIBRATION_RIGHT", 
                                                                    "save_calibration_params_path")
                                                                           )
            msg = f"Read CALIBRATION_RIGHT - save_calibration_params_path: " +\
                  f"{self.right_camera_calibration.save_calibration_params_path}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params_path' in " +\
                          f"section 'CALIBRATION_RIGHT': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.parameters.video_calibration_step = int(config.get("PARAMETERS",
                                                                    "video_calibration_step"))
                                                                     
            msg = f"Read PARAMETERS - video_calibration_step: " +\
                  f"{self.parameters.video_calibration_step}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'video_calibration_step' in " +\
                          f"section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.parameters.num_disparities = int(config.get("PARAMETERS", "num_disparities"))
                                                                     
            msg = f"Read PARAMETERS - num_disparities: {self.parameters.num_disparities}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'num_disparities' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.parameters.block_size = int(config.get("PARAMETERS", "block_size"))
                                                                     
            msg = f"Read PARAMETERS - block_size: {self.parameters.block_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'block_size' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.parameters.gaussian_kernel_size = int(config.get("PARAMETERS", 
                                                                  "gaussian_kernel_size"))
                                                                     
            msg = f"Read PARAMETERS - gaussian_kernel_size: {self.parameters.gaussian_kernel_size}."
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gaussian_kernel_size' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}.")
        return res

        