"""
Implements base class for NavDataEstimator classes.
"""

import logging


__author__ = "EnriqueMoran"


class BaseClass():
    """
    This is the base class for NavDataEstimator classes.
    Its purpose is to set up a common logger for all classes.
    
    Args:
        - format (logging.Formatter): Logger format.
        - filename (str): Path to store log file.
        - level (str): Logger level.
    """

    def __init__(self, filename:str, format:logging.Formatter, level:str):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger