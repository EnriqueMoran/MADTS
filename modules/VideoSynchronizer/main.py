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


gopro_logger = logging.getLogger("OpenGoPro")


class MainApp:
    """
    This class executes the main loop of VideoSynchronizer module.

    Args:
        - args (argparse.Namespace): Arguments to update loggers (Main App and GoPro) parameters. 
        - gopro_filepath (str): Default path to store GoPro messages log file.
        - log_filepath (str): Default path to store Main App messages log file.
        - log_format (str): Main App default logger format.
        - log_level (str): Main App default logger level.
    """

    def __init__(self, log_format:str, log_level:str, log_filepath:str, 
                 gopro_filepath:str, config_filepath:str, args:argparse.Namespace):
        self.gopro_filepath = gopro_filepath
        self.log_filepath   = log_filepath
        self.log_format     = log_format
        self.log_level      = log_level
        self.logger         = None

        self.config_filepath = config_filepath

        self._check_args(args)
        self.set_loggers()


    def _check_args(self, args) -> None:
        """
        Check passed args.

        Args:
            - args (argparse.Namespace): Passed args to be checked.
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
        
        if not args.keep_logs:
            if os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w'):
                    pass

            if os.path.exists(self.gopro_filepath):
                with open(self.gopro_filepath, 'w'):
                    pass


    def set_loggers(self) -> None:
        """
        Create two loggers, one for Main App and another one for GoPro messages.
        """
        global gopro_logger

        gopro_filename = self.gopro_filepath

        # Gopro Logger
        gopro_file_handler = logging.FileHandler(gopro_filename)

        gopro_console_handler = logging.StreamHandler()
        gopro_console_handler.setLevel(logging.ERROR)

        gopro_logger.addHandler(gopro_file_handler)
        gopro_logger.addHandler(gopro_console_handler)
        gopro_logger = setup_logging(base=gopro_logger, output=self.gopro_filepath)


    async def run(self):
        await self.test()


    async def test(self):
        video_syn = VideoSynchronizer(filename=self.log_filepath, format=self.log_format, 
                                      level=self.log_level, config_path=self.config_filepath)
        await video_syn.gopro_manager.start_streaming()


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

    parser.add_argument("--keep_logs", 
                        type=bool,
                        help="If enabled, won't clear logs files on new run.")

    args = parser.parse_args()

    gopro_filepath = f"./modules/VideoSynchronizer/logs/gopro/{datetime.now().strftime('%Y%m%d')}_gopro.log"
    log_filepath   = f"./modules/VideoSynchronizer/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format     = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level      = os.environ.get("LOGLEVEL", "INFO")

    config_filepath = f"./modules/VideoSynchronizer/cfg/config.ini"
    
    app = MainApp(gopro_filepath=gopro_filepath, log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    asyncio.run(app.run())