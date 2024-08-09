"""
TBD
"""

import argparse
import asyncio
import logging
import os

from datetime import datetime
from open_gopro.logger import setup_logging
from pathlib import Path
from src.videosynchronizer import VideoSynchronizer


__author__ = "EnriqueMoran"


logger = logging.getLogger("Main")
gopro_logger = logging.getLogger("OpenGoPro")


class MainApp:
    """
    TBD
    """

    def __init__(self, args):
        self.log_format = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
        self.log_level  = os.environ.get("LOGLEVEL", "INFO")
        self.log_filepath   = f"./logs/{datetime.now().strftime('%Y%m%d')}.log"
        self.gopro_filepath = f"./logs/gopro/{datetime.now().strftime('%Y%m%d')}_gopro.log"
        self.logger = None

        self._check_args(args)
        self.set_loggers()


    def _check_args(self, args) -> None:
        """
        Check passed args.
        """
        valid_log_levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        if args.level:
            if args.level.upper() in valid_log_levels:
                self.log_level = os.environ.get("LOGLEVEL", args.level.upper())
            else:
                message = f"Warning: logging level {args.level} not found within valid values: " +\
                          f"{valid_log_levels}. Defaulting to {self.log_level}."
                print(message)

        if args.log:
            log_path = Path(args.log)
            if log_path.exists():
                self.log_filepath = args.log
            else:
                message = f"Warning: logging file path {log_path} not found. " +\
                        f"Defaulting to {self.log_filepath}."
                print(message)

        if args.gopro_log:
            log_path = Path(args.gopro_log)
            if log_path.exists():
                self.gopro_filepath = args.gopro_log
            else:
                message = f"Warning: GoPro logging file path {log_path} not found. " +\
                        f"Defaulting to {self.gopro_filepath}."
                print(message)


    def set_loggers(self) -> None:
        """
        TBD
        """
        global gopro_logger, logger

        filename = self.log_filepath
        format   = self.log_format
        level    = self.log_level

        # Main app Logger
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger

        # Gopro Logger
        gopro_file_handler = logging.FileHandler(filename)
        gopro_file_handler.setLevel(level)

        gopro_console_handler = logging.StreamHandler()
        gopro_console_handler.setLevel(logging.ERROR)

        gopro_logger.addHandler(gopro_file_handler)
        gopro_logger.addHandler(gopro_console_handler)
        gopro_logger = setup_logging(base=gopro_logger, output=self.gopro_filepath)
        

    async def run(self):
        await self.test()

    async def test(self):
        video_syn = VideoSynchronizer(filename=self.log_filepath, format=self.log_format, 
                                      level=self.log_level, config_path=self.gopro_filepath)
        await video_syn.gopro_manager.connect_cameras()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADTS Video Synchronizer.")

    parser.add_argument("--level", 
                        type=str,
                        help="Set loggin level. Valid values: DEBUG, INFO, WARN, ERROR, CRITICAL.")

    parser.add_argument("--log", 
                        type=str,
                        help="Set logging file path.")
    
    parser.add_argument("--gopro_log", 
                        type=str,
                        help="Set GoPro logging file path.")

    args = parser.parse_args()
    app = MainApp(args)

    asyncio.run(app.run())