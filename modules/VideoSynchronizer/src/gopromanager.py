"""
TBD

WirelessGoPro documentation: https://github.com/gopro/OpenGoPro/tree/main/demos/python/sdk_wireless_camera_control/docs
OpenGoPro usage: https://github.com/gopro/OpenGoPro/blob/78a3864e2447f2eb631ed5b730891e42fe40457b/demos/python/sdk_wireless_camera_control/docs/usage.rst
OpenGoPro Python SDK: https://gopro.github.io/OpenGoPro/python_sdk/api.html
"""

import asyncio
import logging

from open_gopro import Params, WirelessGoPro
from open_gopro.constants import StatusId
from src.baseclass import BaseClass
from src.configmanager import ConfigManager
from src.utils.enums import GoProAction



__author__ = "EnriqueMoran"


class GoProManager(BaseClass):
    """
    TBD
 
    +---------------+       +-------------+ +-------------+
    | GoProManager  |       | Left GoPro  | | Right GoPro |
    +---------------+       +-------------+ +-------------+
            |                      |               |
            | Connect              |               |
            |--------------------->|               |
            |                      |               |
            |                   OK |               |
            |<---------------------|               |
            |                      |               |
            | Get statuses         |               |
            |--------------------->|               |
            |                      |               |
            |             Statuses |               |
            |<---------------------|               |
            |                      |               |
            | Connect              |               |
            |------------------------------------->|
            |                      |               |
            |                      |            OK |
            |<-------------------------------------|
            |                      |               |
            | Get statuses         |               |
            |------------------------------------->|
            |                      |               |
            |                      |      Statuses |
            |<-------------------------------------|
            |                      |               |
            | Stream ready?        |               |
            |--------------------->|               |
            |                      |               |
            |                  YES |               |
            |<---------------------|               |
            |                      |               |
            | Stream ready?        |               |
            |------------------------------------->|
            |                      |               |
            |                      |           YES |
            |<-------------------------------------|
            |                      |               |
            | Start stream         |               |
            |--------------------->|               |
            |                      |               |
            | Start stream         |               |
            |------------------------------------->|
            |                      |               |
            |                   OK |               |
            |<---------------------|               |
            |                      |               |
            |                      |            OK |
            |<-------------------------------------|
            |                      |               |
            |         ...          |     ...       |
            |                      |               |
            | Get statuses         |               |
            |--------------------->|               |
            |                      |               |
            |             Statuses |               |
            |<---------------------|               |
            |                      |               |
            | Get statuses         |               |
            |------------------------------------->|
            |                      |               |
            |                      |      Statuses |
            |<-------------------------------------|
            |                      |               |
            |         ...          |     ...       |
            |                      |               |
            | Stop stream          |               |
            |--------------------->|               |
            |                      |               |
            | Stop stream          |               |
            |------------------------------------->|
            |                      |               |
            |                   OK |               |
            |<---------------------|               |
            |                      |               |
            |                      |            OK |
            |<-------------------------------------|
            |                      |               | 
    """

    def __init__(self, filename:str, format:logging.Formatter, level:str, 
                 config_manager: ConfigManager, record_stream:bool = False):
        super().__init__(filename, format, level)
        self.gopro_right_name = config_manager.gopro_right_name
        self.gopro_left_name  = config_manager.gopro_left_name
        self.record_stream    = config_manager.record_stream  # TODO
        self.use_http         = config_manager.use_http
    

    async def perform_actions(self, gopro:WirelessGoPro, queue:asyncio.Queue) -> None:
        """
        TBD
        """
        while not queue.empty():
            action = await queue.get()
            command_type = "WIFI" if self.use_http else "BLE"
            target = gopro.identifier

            # Retrieve GoPro status to check everything is ok
            if action == GoProAction.GET_STATUS:
                if self.use_http:
                    log_msg = f"Sending get_camera_state command through {command_type} to {target}"
                    statuses = await gopro.http_command.get_camera_state()
                else:
                    log_msg = f"Sending get_camera_statuses command through {command_type} to {target}"
                    statuses = await gopro.ble_command.get_camera_statuses()
            
                self.logger.info(log_msg)
                self.logger.debug(f"Received status data from {target}: {statuses}")

                remaining_video_time = int(statuses.data[StatusId.VIDEO_REM])
                primary_storage      = str(statuses.data[StatusId.SD_STATUS])
                pairing_state        = str(statuses.data[StatusId.PAIR_STATE])
                overheating          = bool(statuses.data[StatusId.SYSTEM_HOT])
                zoom_level           = int(statuses.data[StatusId.DIGITAL_ZOOM])
                wifi_bars            = int(statuses.data[StatusId.WIFI_BARS])
                lens_type            = str(statuses.data[StatusId.CAMERA_LENS_TYPE])
                ap_ssid              = str(statuses.data[StatusId.AP_SSID])
                battery              = int(statuses.data[StatusId.INT_BATT_PER])
                is_busy              = bool(statuses.data[StatusId.SYSTEM_BUSY])

                video_hours   = remaining_video_time // 3600
                video_minutes = (remaining_video_time % 3600) // 60
                video_seconds = remaining_video_time % 60
                remaining_video_time_msg = f"{video_hours:02}:{video_minutes:02}:{video_seconds:02}"

                self.logger.info(f"Received status data from {target}:")
                self.logger.info(f"\tAP SSID: {ap_ssid}")
                self.logger.info(f"\tBattery level: {battery}%")
                self.logger.info(f"\tWifi strength: {wifi_bars}")
                self.logger.info(f"\tIs busy: {is_busy}")
                self.logger.info(f"\tIs overheating: {overheating}")
                self.logger.info(f"\tPairing state: {pairing_state}")
                self.logger.info(f"\tLens type: {lens_type}")
                self.logger.info(f"\tZoom level: {zoom_level}%")
                self.logger.info(f"\tPrimary storage: {primary_storage}")
                self.logger.info(f"\tRemaining video time: {remaining_video_time_msg}")
                # Notify finished task
                queue.task_done()
            
            elif action == GoProAction.ENABLE_WIFI:
                log_msg = f"Sending enable_wifi_ap(True) command through BLE to {target}"
                self.logger.info(log_msg)
                response_enable_wifi = await gopro.ble_command.enable_wifi_ap(enable=True)

                self.logger.info(f"Received response data from {target}:")
                self.logger.info(f"\tid: {response_enable_wifi['id']}")
                self.logger.info(f"\tstatus: {response_enable_wifi['status']}")
                self.logger.info(f"\tprotocol: {response_enable_wifi['protocol']}")
                # Notify finished task
                queue.task_done()

            elif action == GoProAction.CONNECT_WIFI:
                log_msg = f"Sending request_wifi_connect_new(True) command through BLE to {target}"
                self.logger.info(log_msg)
                request_response = await gopro.ble_command.request_wifi_connect_new()
                self.logger.info(f"Received response data from {target}:")
                self.logger.info(f"{request_response}")
                # Notify finished task
                queue.task_done()
    

    async def manage_gopros(self):
        """
        TBD
        """
        gopro_left_queue  = asyncio.Queue()
        gopro_right_queue = asyncio.Queue()

        # Connect to LEFT GO PRO
        async with WirelessGoPro(target=self.gopro_left_name, enable_wifi=False) as gopro_left:
            self.logger.info(f"Connected to LEFT GO PRO! ({gopro_left.identifier})")

            left_task = asyncio.create_task(self.perform_actions(gopro_left,
                                                                  gopro_left_queue))

            # 1. Retrieve GoPro status for the first time to check everything is ok
            await gopro_left_queue.put(GoProAction.GET_STATUS)
            # 2. If configured, enable WIFI communication
            if self.use_http:
                await gopro_left_queue.put(GoProAction.ENABLE_WIFI)
                await gopro_left_queue.put(GoProAction.CONNECT_WIFI)

            # Connect to RIGHT GO PRO once connection to LEFT GO PRO has been successful
            async with WirelessGoPro(target=self.gopro_right_name, enable_wifi=False) as gopro_right:
                self.logger.info(f"Connected to RIGHT GO PRO! ({gopro_right.identifier})")

                right_task = asyncio.create_task(self.perform_actions(gopro_right, 
                                                                     gopro_right_queue))

                # 1. Retrieve GoPro status for the first time to check everything is ok
                await gopro_right_queue.put(GoProAction.GET_STATUS)
                # 2. If configured, enable WIFI communication
                if self.use_http:
                    await gopro_right_queue.put(GoProAction.ENABLE_WIFI)
                    await gopro_right_queue.put(GoProAction.CONNECT_WIFI)


                await right_task    # End connection with RIGHT GO PRO
            self.logger.info(f"Finished connection to RIGHT GO PRO!")

            await left_task    # End connection with LEFT GO PRO
            self.logger.info(f"Finished connection to LEFT GO PRO")
        
        await gopro_left_queue.join()
        await gopro_right_queue.join()
    

    async def start_streaming(self) -> None:
        """
        TBD
        """
        await self.manage_gopros()

        

    async def record_left_camera(self, event:asyncio.Event) -> None:
        """
        TBD
        """
        self.logger.info(f"Trying to connect to left GoPro...")
        try:
            async with WirelessGoPro(target=self.gopro_left_name) as gopro_left:
                self.logger.info(f"Connected to left GoPro!")
                event.set()
                await gopro_left.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
                await asyncio.sleep(15)
                await gopro_left.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        except OSError as e:
            self.logger.error(f"OS Error raised: {e}")


    async def record_right_camera(self, event:asyncio.Event) -> None:
        """
        TBD
        """
        self.logger.info(f"Trying to connect to right GoPro...")
        self.logger.info(f"Waiting for left camera to connect...")
        await event.wait()
        try:
            async with WirelessGoPro(target=self.gopro_right_name) as gopro_right:
                self.logger.info(f"Connected to right GoPro!")
                await gopro_right.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
                await asyncio.sleep(15)
                await gopro_right.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        except OSError as e:
            self.logger.error(f"OS Error raised: {e}")
        
    
    