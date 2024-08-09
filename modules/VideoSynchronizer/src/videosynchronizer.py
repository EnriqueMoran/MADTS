"""
TBD
"""

import logging

from src.baseclass import BaseClass
from src.configmanager import ConfigManager
from src.gopromanager import GoProManager


__author__ = "EnriqueMoran"


class VideoSynchronizer(BaseClass):
    """
    TBD
    """

    def __init__(self, filename:str, format:logging.Formatter, level:str, 
                 config_path="./cfg/config.ini"):
        super().__init__(filename, format, level)
        self.config_parser = ConfigManager(filename=filename, format=format, level=level, 
                                           config_path=config_path)
        self.gopro_manager = GoProManager(filename=filename, format=format, level=level,
                                          config_manager=self.config_parser)

        self.config_parser.read_config()
    
