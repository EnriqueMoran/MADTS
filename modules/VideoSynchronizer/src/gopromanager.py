"""
TBD

WirelessGoPro documentation: https://github.com/gopro/OpenGoPro/tree/main/demos/python/sdk_wireless_camera_control/docs
OpenGoPro Python SDK: https://gopro.github.io/OpenGoPro/python_sdk/api.htm
"""

import asyncio
import logging

from open_gopro.logger import setup_logging
from open_gopro import Params, WirelessGoPro
from src.configmanager import ConfigManager

import time


__author__ = "EnriqueMoran"


logger = logging.getLogger("GoProManager")
gopro_logger = None


class GoProManager():
    """
    TBD
    """

    def __init__(self, config_manager: ConfigManager, gopro_logger_path):
        self.gopro_right_name = config_manager.gopro_right_name
        self.gopro_left_name = config_manager.gopro_left_name
        self.set_gopro_logger(gopro_logger_path)

    
    def set_gopro_logger(self, gopro_logger_path: str) -> None:
        """
        TBD
        """
        global gopro_logger
        gopro_logger = setup_logging("GoProManager", gopro_logger_path)


    async def record_left_camera(self, event:asyncio.Event) -> None:
        """
        TBD
        """
        logger.info(f"Trying to connect to left GoPro...")
        try:
            async with WirelessGoPro(target=self.gopro_left_name) as gopro_left:
                logger.info(f"Connected to left GoPro!")
                event.set()
                await gopro_left.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
                await asyncio.sleep(15)
                await gopro_left.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        except OSError as e:
            logger.error(f"OS Error raised: {e}")


    async def record_right_camera(self, event:asyncio.Event) -> None:
        """
        TBD
        """
        logger.info(f"Trying to connect to right GoPro...")
        logger.info(f"Waiting for left camera to connect...")
        await event.wait()
        try:
            async with WirelessGoPro(target=self.gopro_right_name) as gopro_right:
                logger.info(f"Connected to right GoPro!")
                await gopro_right.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
                await asyncio.sleep(15)
                await gopro_right.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        except OSError as e:
            logger.error(f"OS Error raised: {e}")
        
    
    async def connect_cameras(self) -> None:
        """
        TBD
        """
        left_camera_connected = asyncio.Event()

        await asyncio.gather(
            self.record_left_camera(left_camera_connected),
            self.record_right_camera(left_camera_connected)
        )