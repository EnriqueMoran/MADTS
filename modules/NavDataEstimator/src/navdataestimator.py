"""
Implements main class.
"""

from src.baseclass import BaseClass
from src.configmanager import ConfigManager


__author__ = "EnriqueMoran"


class NavDataEstimator(BaseClass):
    """
    This is the main class, its in charge of estimating navigation data (distance, heading and 
    speed) from camera's stream.

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
        
    
