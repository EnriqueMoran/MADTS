"""
Implements configuration parser class.
"""

import configparser

from pathlib import Path

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.utils.enums import CalibrationMode, RectificationMode, \
                                                     UndistortMethod
from modules.NavDataEstimator.src.utils.globals import *
from modules.NavDataEstimator.src.utils.structs import *


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
            self.left_camera_specs = CameraSpecs()
            
            ##### RIGHT_CAMERA #####
            self.right_camera_specs = CameraSpecs()

            ##### SYSTEM_SETUP #####
            self.system_setup = SystemSetup()

            ##### CALIBRATION #####
            self.calibration = CalibrationSpecs()

            ##### PARAMETERS #####
            self.parameters = Parameters()

            #### STREAM ####
            self.stream = Stream()

            #### COMMUNICATION IN ####
            self.comm_in = Communication()

            #### COMMUNICATION OUT ####
            self.comm_out = Communication()

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
            self.left_camera_specs.focal_length = float(config.get(
                                                            "LEFT_CAMERA_SPECS", 
                                                            "focal_length").strip()
                                                        )
            msg = f"Read LEFT_CAMERA - focal_length: {self.left_camera_specs.focal_length}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_camera_specs.pixel_size = float(config.get(
                                                          "LEFT_CAMERA_SPECS", 
                                                          "pixel_size").strip()
                                                     )
            msg = f"Read LEFT_CAMERA - pixel_size: {self.left_camera_specs.pixel_size}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.left_camera_specs.fov = float(config.get(
                                                   "LEFT_CAMERA_SPECS", 
                                                   "horizontal_fov").strip()
                                              )
            msg = f"Read LEFT_CAMERA - horizontal_fov: {self.left_camera_specs.fov}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'LEFT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.right_camera_specs.focal_length = int(config.get(
                                                           "RIGHT_CAMERA_SPECS", 
                                                           "focal_length").strip()
                                                      )
            msg = f"Read RIGHT_CAMERA - focal_length: {self.right_camera_specs.focal_length}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'focal_length' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_camera_specs.pixel_size = float(config.get(
                                                           "RIGHT_CAMERA_SPECS", 
                                                           "pixel_size").strip()
                                                      )
            msg = f"Read RIGHT_CAMERA - pixel_size: {self.right_camera_specs.pixel_size}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'pixel_size' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.right_camera_specs.fov = float(config.get(
                                                    "RIGHT_CAMERA_SPECS", 
                                                    "horizontal_fov").strip()
                                               )
            msg = f"Read RIGHT_CAMERA - horizontal_fov: {self.right_camera_specs.fov}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'horizontal_fov' in section 'RIGHT_CAMERA': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.system_setup.baseline_distance = float(config.get(
                                                            "SYSTEM_SETUP", 
                                                            "baseline_distance").strip()
                                                       )
            msg = f"Read SYSTEM_SETUP - baseline_distance: {self.system_setup.baseline_distance}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'baseline_distance' in section 'SYSTEM_SETUP': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.calibration.chessboard_width = int(config.get(
                                                        "CALIBRATION",
                                                        "chessboard_width").strip()
                                                   )
            msg = f"Read CALIBRATION - chessboard_width: " +\
                  f"{self.calibration.chessboard_width}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_width' in section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.chessboard_height = int(config.get(
                                                        "CALIBRATION", 
                                                        "chessboard_height").strip()
                                                    )
            msg = f"Read CALIBRATION - chessboard_height: " +\
                  f"{self.calibration.chessboard_height}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_height' in section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.chessboard_square_size = int(config.get(
                                                              "CALIBRATION",
                                                              "chessboard_square_size").strip()
                                                         )
            msg = f"Read CALIBRATION - chessboard_square_size: " +\
                  f"{self.calibration.chessboard_square_size}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'chessboard_square_size' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            value = int(config.get("CALIBRATION", "calibration_mode").strip())
                                                          
            msg = f"Read CALIBRATION - calibration_mode: {value}"
            self.logger.info(msg)

            valid_values = CalibrationMode._value2member_map_
            if value not in valid_values:
                warning_msg = f"Value not valid, accepted values': {valid_values}"
                self.logger.warning(warning_msg)

                value = DEFAULT_CALIBRATION_MODE
                warning_msg = f"Value set as {value}"
                self.logger.warning(warning_msg)
            self.calibration.calibration_mode = CalibrationMode(value)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'calibration_mode' in section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.save_calibration_images = bool(int(config.get(
                                                                    "CALIBRATION", 
                                                                    "save_calibration_images")
                                                                    .strip())
                                                           )
            msg = f"Read CALIBRATION - save_calibration_images: " +\
                  f"{self.calibration.save_calibration_images}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_images' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.save_calibration_params = bool(int(config.get(
                                                                    "CALIBRATION", 
                                                                    "save_calibration_params")
                                                                    .strip())
                                                           )
            msg = f"Read CALIBRATION - save_calibration_params: " +\
                  f"{self.calibration.save_calibration_params}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.video_calibration_step = int(config.get(
                                                             "CALIBRATION",
                                                             "video_calibration_step").strip()
                                                        )
                                                                     
            msg = f"Read CALIBRATION - video_calibration_step: " +\
                  f"{self.calibration.video_calibration_step}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'video_calibration_step' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.save_calibration_images_path = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "save_calibration_images_path")
                                                                    .strip()
                                                               )
            msg = f"Read CALIBRATION - save_calibration_images_path: " +\
                  f"{self.calibration.save_calibration_images_path}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_images_path' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.calibration_images_dir_left = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "calibration_images_dir_left")
                                                                    .strip()
                                                               )
            msg = f"Read CALIBRATION - calibration_images_dir_left: " +\
                  f"{self.calibration.calibration_images_dir_left}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'calibration_images_dir_left' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.calibration.calibration_images_dir_right = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "calibration_images_dir_right")
                                                                    .strip()
                                                               )
            msg = f"Read CALIBRATION - calibration_images_dir_right: " +\
                  f"{self.calibration.calibration_images_dir_right}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'calibration_images_dir_right' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.calibration_video_left = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "calibration_video_left")
                                                                    .strip()
                                                         )
            msg = f"Read CALIBRATION - calibration_video_left: " +\
                  f"{self.calibration.calibration_video_left}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'calibration_video_left' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.calibration_video_right = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "calibration_video_right")
                                                                    .strip()
                                                          )
            msg = f"Read CALIBRATION - calibration_video_right: " +\
                  f"{self.calibration.calibration_video_right}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'calibration_video_right' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.save_calibration_params_path = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "save_calibration_params_path")
                                                                    .strip()
                                                               )
            msg = f"Read CALIBRATION - save_calibration_params_path: " +\
                  f"{self.calibration.save_calibration_params_path}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'save_calibration_params_path' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.calibration.load_calibration_params_path = str(config.get(
                                                                    "CALIBRATION", 
                                                                    "load_calibration_params_path")
                                                                    .strip()
                                                               )
            msg = f"Read CALIBRATION - load_calibration_params_path: " +\
                  f"{self.calibration.load_calibration_params_path}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'load_calibration_params_path' in " +\
                          f"section 'CALIBRATION': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            value = int(config.get(
                            "PARAMETERS", 
                            "num_disparities").strip()
                       )
            
            msg = f"Read PARAMETERS - num_disparities: {value}"
            self.logger.info(msg)

            if value % 16 != 0:
                remainder = value % 16
                if remainder < 8:
                    value = value - remainder
                else:
                    value = value + (16 - remainder)
                warning_msg = f"Value is not multiple of 16, set as {value}"
                self.logger.warning(warning_msg)

            self.parameters.num_disparities = value
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'num_disparities' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            value = int(config.get(
                            "PARAMETERS", 
                            "block_size").strip()
                       )
                                                                     
            msg = f"Read PARAMETERS - block_size: {value}"
            self.logger.info(msg)

            if value % 2 == 0:
                value += 1
                warning_msg = f"Value is not odd, set as {value}"
                self.logger.warning(warning_msg)

            self.parameters.block_size = value
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'block_size' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.parameters.gaussian_kernel_size = int(config.get(
                                                           "PARAMETERS", 
                                                           "gaussian_kernel_size".strip())
                                                      )
                                                                     
            msg = f"Read PARAMETERS - gaussian_kernel_size: {self.parameters.gaussian_kernel_size}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'gaussian_kernel_size' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            value = float(config.get(
                              "PARAMETERS", 
                              "rectify_alpha").strip()
                         )
            
            msg = f"Read PARAMETERS - rectify_alpha: {value}"
            self.logger.info(msg)
            
            if MIN_ALPHA > value:
                value = MIN_ALPHA
                warning_msg = f"Value lower than minimum ({MIN_ALPHA}), value set as 0."
                self.logger.warning(warning_msg)
            elif value > MAX_ALPHA:
                value = MAX_ALPHA
                warning_msg = f"Value higher than maximum ({MAX_ALPHA}), value set as 1."
                self.logger.warning(warning_msg)

            self.parameters.rectify_alpha = value                                              
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'rectify_alpha' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            value = int(config.get(
                            "PARAMETERS", 
                            "undistort_method").strip()
                       )
                                                          
            msg = f"Read PARAMETERS - undistort_method: {value}"
            self.logger.info(msg)

            valid_values = UndistortMethod._value2member_map_
            if value not in valid_values:
                warning_msg = f"Value not valid, accepted values': {valid_values}"
                self.logger.warning(warning_msg)

                value = DEFAULT_UNDISTORT_METHOD
                warning_msg = f"Value set as {value}"
                self.logger.warning(warning_msg)
            self.parameters.undistort_method = UndistortMethod(value)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'undistort_method' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            value = int(config.get(
                            "PARAMETERS", 
                            "rectification_mode").strip()
                       )
                                                          
            msg = f"Read PARAMETERS - rectification_mode: {value}"
            self.logger.info(msg)

            valid_values = RectificationMode._value2member_map_
            if value not in valid_values:
                warning_msg = f"Value not valid, accepted values': {valid_values}"
                self.logger.warning(warning_msg)

                value = DEFAULT_RECTIFICATION_MODE
                warning_msg = f"Value set as {value}"
                self.logger.warning(warning_msg)
            self.parameters.rectification_mode = RectificationMode(value)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'rectification_mode' in section 'PARAMETERS': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.stream.left_camera = str(config.get(
                                              "STREAM", 
                                              "left_camera_url").strip()
                                         )
                                                                     
            msg = f"Read STREAM - left_camera_url: {self.stream.left_camera}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'left_camera_url' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.stream.right_camera = str(config.get(
                                               "STREAM", 
                                               "right_camera_url").strip()
                                          )   
                                                                     
            msg = f"Read STREAM - right_camera_url: {self.stream.right_camera}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'right_camera_url' in section 'STREAM': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.comm_in.group = str(config.get(
                                         "COMMUNICATION_IN", 
                                         "multicast_group").strip()
                                    )
                                                                     
            msg = f"Read COMMUNICATION_IN - multicast_group: {self.comm_in.group}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_group' in section 'COMMUNICATION_IN': {e}"
            self.logger.warning(warning_msg)
            res = False
        
        try:
            self.comm_in.port = int(config.get(
                                        "COMMUNICATION_IN", 
                                        "multicast_port").strip()
                                   )

            msg = f"Read COMMUNICATION_IN - multicast_port: {self.comm_in.port}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_port' in section 'COMMUNICATION_IN': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.comm_in.iface = str(config.get(
                                         "COMMUNICATION_IN", 
                                         "multicast_iface").strip()
                                    )
                                                                     
            msg = f"Read COMMUNICATION_IN - multicast_iface: {self.comm_in.iface}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_iface' in section 'COMMUNICATION_IN': {e}"
            self.logger.warning(warning_msg)
            res = False

        try:
            self.comm_in.ttl = int(config.get(
                                       "COMMUNICATION_IN", 
                                       "multicast_ttl").strip()
                                   )

            msg = f"Read COMMUNICATION_IN - multicast_ttl: {self.comm_in.ttl}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'multicast_ttl' in section 'COMMUNICATION_IN': {e}"
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
            self.comm_out.frequency = float(config.get(
                                                "COMMUNICATION_OUT", 
                                                "send_frequency").strip()
                                           )

            msg = f"Read COMMUNICATION_OUT - send_frequency: {self.comm_out.frequency}"
            self.logger.info(msg)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            warning_msg = f"Could not find 'send_frequency' in section 'COMMUNICATION_OUT': {e}"
            self.logger.warning(warning_msg)
            res = False


        self.logger.info(f"Finished reading configuration, all params read successfully: {res}")
        return res
        
        