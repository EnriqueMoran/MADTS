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

        self.logger.info(f"Finished reading configuration, all params read successfully: {res}")
        return res
        
        