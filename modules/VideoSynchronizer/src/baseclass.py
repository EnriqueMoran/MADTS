"""
TBD
"""

import logging


class BaseClass():
    def __init__(self, filename:str, format:logging.Formatter, level:str):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(self.__class__.__name__)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger