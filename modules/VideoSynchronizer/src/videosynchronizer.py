"""
Implements main class.
"""

from modules.VideoSynchronizer.src.baseclass import BaseClass
from modules.VideoSynchronizer.src.configmanager import ConfigManager
from modules.VideoSynchronizer.src.gopromanager import GoProManager


__author__ = "EnriqueMoran"


class VideoSynchronizer(BaseClass):
    """
    This is the main class, its in charge of receiving video streams from cameras, synchronizing
    them and send them again.

    Args:
        - config_path (str): Path to configuration file.
        - filename (str): Path to store log file; belongs to BaseClass.
        - format (str): Logger format; belongs to BaseClass.
        - level (str): Logger level; belongs to BaseClass.
    """

    def __init__(self, filename:str, format:str, level:str, config_path:str):
        super().__init__(filename, format, level)
        self.config_parser = ConfigManager(filename=filename, format=format, level=level, 
                                           config_path=config_path)
        self.gopro_manager = GoProManager(filename=filename, format=format, level=level,
                                          config_manager=self.config_parser)
    
