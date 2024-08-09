"""
TBD
"""

import logging

from src.configmanager import ConfigManager
from src.gopromanager import GoProManager


__author__ = "EnriqueMoran"


logger = logging.getLogger("VideoSynchronizer")


class VideoSynchronizer():
    """
    TBD
    """

    def __init__(self, gopro_logger_path, config_path="./cfg/config.ini"):
        self.config_parser = ConfigManager(config_path)
        self.gopro_manager = GoProManager(self.config_parser, gopro_logger_path)

        self.config_parser.read_config()
    
