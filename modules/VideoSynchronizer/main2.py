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
gopro_logger = None

class MainApp:
    """
    TBD
    """

    def __init__(self, args):
        self.log_format = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
        self.log_level = os.environ.get("LOGLEVEL", "INFO")
        self.filepath = f"./logs/{datetime.now().strftime('%Y%m%d')}.log"

        self.gopro_filepath = f"./logs/gopro/{datetime.now().strftime('%Y%m%d')}_gopro.log"

        self._check_args(args)
        self.set_logger()


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
                self.filepath = args.log
            else:
                message = f"Warning: logging file path {log_path} not found. " +\
                        f"Defaulting to {self.filepath}."
                print(message)
        
        if args.gopro_log:
            log_path = Path(args.gopro_log)
            if log_path.exists():
                self.gopro_filepath = args.gopro_log
            else:
                message = f"Warning: GoPro logging file path {log_path} not found. " +\
                        f"Defaulting to {self.gopro_filepath}."
                print(message)

    
    def set_logger(self) -> None:
        """
        TBD
        """
        filename = self.filepath
        format   = self.log_format
        level    = self.log_level
        logging.basicConfig(format=format, level=level, filename=filename)
    

    async def run(self):
        await self.test()

    async def test(self):
        video_syn = VideoSynchronizer(self.gopro_filepath)
        await video_syn.gopro_manager.connect_right_camera()


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