"""
Implements main class.
"""

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager
from modules.NavDataEstimator.src.distanceEstimator.distancecalculator import DistanceCalculator


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
        self.config_parser = ConfigManager(filename=filename, 
                                           format=format, 
                                           level=level, 
                                           config_path=config_path)
        
        self.distance_calculator = DistanceCalculator(filename=filename, 
                                                      format=format, 
                                                      level=level, 
                                                      config_path=config_path)