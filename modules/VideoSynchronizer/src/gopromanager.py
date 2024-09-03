"""
TBD

WirelessGoPro documentation: https://github.com/gopro/OpenGoPro/tree/main/demos/python/sdk_wireless_camera_control/docs
OpenGoPro usage: https://github.com/gopro/OpenGoPro/blob/78a3864e2447f2eb631ed5b730891e42fe40457b/demos/python/sdk_wireless_camera_control/docs/usage.rst
OpenGoPro Python SDK: https://gopro.github.io/OpenGoPro/python_sdk/api.html
"""

import asyncio

from open_gopro import Params, WirelessGoPro, proto
from open_gopro.constants import StatusId, ActionId
from open_gopro.exceptions import FailedToFindDevice

from typing import Any

from modules.VideoSynchronizer.src.baseclass import BaseClass
from modules.VideoSynchronizer.src.configmanager import ConfigManager
from modules.VideoSynchronizer.src.utils.enums import GoProAction


__author__ = "EnriqueMoran"


class GoProManager(BaseClass):
    """
    TBD
    """

    def __init__(self, filename:str, format:str, level:str, config_manager: ConfigManager):
        super().__init__(filename, format, level)
        self.gopro_right_name    = config_manager.gopro_right_name
        self.gopro_left_name     = config_manager.gopro_left_name
        self.gopro_right_in_url  = config_manager.gopro_right_in_url
        self.gopro_left_in_url   = config_manager.gopro_left_in_url

        self.network_ssid = config_manager.network_ssid
        self.network_pass = config_manager.network_password

        self.record_stream = config_manager.record_stream  # TODO
        self.min_bitrate   = config_manager.min_bitrate
        self.max_bitrate   = config_manager.max_bitrate
        self.start_bitrate = config_manager.starting_bitrate

        if config_manager.resolution == 0:
            self.resolution = proto.EnumWindowSize.WINDOW_SIZE_480
        elif config_manager.resolution == 1:
            self.resolution = proto.EnumWindowSize.WINDOW_SIZE_720
        else:
            self.resolution = proto.EnumWindowSize.WINDOW_SIZE_1080

        if config_manager.fov == 0:
            self.fov = proto.EnumLens.LENS_WIDE
        elif config_manager.fov == 1:
            self.fov = proto.EnumLens.LENS_LINEAR
        else:
            self.fov = proto.EnumLens.LENS_SUPERVIEW


    async def perform_actions(self, gopro:WirelessGoPro, queue:asyncio.Queue) -> None:
        """
        TBD
        """
        while not queue.empty():
            self.logger.debug(f"Action Queue (size {queue.qsize()}): {queue}")

            action = await queue.get()
            target = gopro.identifier

            # Retrieve GoPro status to check everything is ok
            if action == GoProAction.GET_STATUS_BLE:
                log_msg = f"Sending get_camera_statuses command through BLE to {target}."
                self.logger.info(log_msg)

                statuses = await gopro.ble_command.get_camera_statuses()                
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
            
            # Connect Go Pro to the same network as RTMP server
            if action == GoProAction.CONECT_TO_NETWORK:
                self.logger.info(f"Connecting {target} to network {self.network_ssid}...")
                log_msg = f"Sending request_wifi_connect_new({self.network_ssid}, " +\
                          f"{self.network_pass}) command through BLE to {target}."
                self.logger.debug(log_msg)
                response = await gopro.ble_command.request_wifi_connect_new(ssid=self.network_ssid, 
                                                                            password=self.network_pass)
                
                self.logger.debug(f"Received response from {target}: {response}")
                response_code = response.data["result"]
                if response_code == proto.EnumResultGeneric.RESULT_SUCCESS:
                    self.logger.info(f"{target} Starting connecting to network...")
                else:
                    self.logger.warning(f"{target} couldn't conbnect to network, code: {response_code}")
                
                network_is_ready = asyncio.Event()
                
                async def wait_for_connecting_to_network(_: Any, update: proto.EnumProvisioning) -> None:
                    self.logger.debug(f"Received update from {target}: {update}")
                    if update['provisioning_state'] == proto.EnumProvisioning.PROVISIONING_SUCCESS_NEW_AP:
                        network_is_ready.set()
                        self.logger.info(f"{target} connected to network!")
                    else:
                        self.logger.warning(f"Couldn't connect to network.")
                    gopro.unregister_update(wait_for_connecting_to_network)

                gopro.register_update(wait_for_connecting_to_network, ActionId.NOTIF_PROVIS_STATE)

                self.logger.info(f"Waiting for {target} to connect to network...")
                await network_is_ready.wait()
                
                # Notify finished task
                queue.task_done()
                
                

            # Configure Go Pro to start streaming
            if action == GoProAction.CONFIGURE_STREAM:
                log_msg = f"Sending set_shutter(DISABLE) command through BLE to {target}."
                self.logger.debug(log_msg)
                response = await gopro.ble_command.set_shutter(shutter=Params.Toggle.DISABLE)
                self.logger.debug(f"Received response from {target}: {response}")

                left_livestream_is_ready  = asyncio.Event()
                right_livestream_is_ready = asyncio.Event()

                gopro_name = target.split(":")[-1].strip()

                await gopro.ble_command.register_livestream_status(
                    register=[proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_STATUS]
                    )
                
                async def wait_for_livestream_start(_: Any, update: proto.NotifyLiveStreamStatus) -> None:
                    self.logger.debug(f"Received update from {target}: {update}")
                    if update.live_stream_status == proto.EnumLiveStreamStatus.LIVE_STREAM_STATE_READY:
                        if gopro_name == self.gopro_left_name:
                            left_livestream_is_ready.set()
                            self.logger.info(f"LEFT GO PRO livestream is ready!")
                            gopro.unregister_update(wait_for_livestream_start)
                        else:
                            right_livestream_is_ready.set()
                            self.logger.info(f"RIGHT GO PRO livestream is ready!")
                            gopro.unregister_update(wait_for_livestream_start)
                        
                self.logger.info(f"Configuring {gopro_name} livestream:")
                url = self.gopro_left_in_url if gopro_name == self.gopro_left_name else self.gopro_right_in_url
                self.logger.info(f"\turl: {url}")
                self.logger.info(f"\twindow_size:       {self.resolution}")
                self.logger.info(f"\tminimum_bitrate:  {self.min_bitrate}")
                self.logger.info(f"\tmaximum_bitrate:  {self.max_bitrate}")
                self.logger.info(f"\tstarting_bitrate: {self.start_bitrate}")
                self.logger.info(f"\tfov:              {self.fov}")

                gopro.register_update(wait_for_livestream_start, ActionId.LIVESTREAM_STATUS_NOTIF)
                await gopro.ble_command.set_livestream_mode(
                    url=url,
                    window_size=self.resolution,
                    minimum_bitrate=self.min_bitrate,
                    maximum_bitrate=self.max_bitrate,
                    starting_bitrate=self.start_bitrate,
                    lens=self.fov,
                )    
                
                self.logger.info(f"Waiting for {gopro_name} livestream to be ready...")
                if gopro_name == self.gopro_left_name:
                    ready_response = await left_livestream_is_ready.wait()
                else:
                    ready_response = await right_livestream_is_ready.wait()

                self.logger.debug(f"Response from {gopro_name}: {ready_response}")
                
                await asyncio.sleep(2)    # In SDK example this is used
                
                self.logger.info(f"{gopro_name} is ready for streaming.")
                # Notify finished task
                queue.task_done()
                
            
            # Start streaming
            if action == GoProAction.START_STREAMING:
                self.logger.info(f"Starting {gopro_name} livestream...")
                log_msg = f"Sending set_shutter(ENABLE) command through BLE to {target}."
                self.logger.debug(log_msg)
                start_response = await gopro.ble_command.set_shutter(shutter=Params.Toggle.ENABLE)
                self.logger.debug(f"Received response from {target}: {start_response}")
                self.logger.info(f"{gopro_name} Livestream on!")

                await asyncio.sleep(60)    # Test stream 1 min

                self.logger.info(f"Closing {gopro_name} livestream...")
                log_msg = f"Sending set_shutter(DISABLE) command through BLE to {target}."
                self.logger.debug(log_msg)
                stop_response = await gopro.ble_command.set_shutter(shutter=Params.Toggle.DISABLE)
                self.logger.debug(f"Received response from {target}: {stop_response}")
                self.logger.info(f"{gopro_name} Livestream ended!")

                # Notify finished task
                queue.task_done()
            


    async def manage_gopros(self):
        """
        TBD
        """
        gopro_left_queue  = asyncio.Queue()
        gopro_right_queue = asyncio.Queue()

        while True:
            self.logger.info(f"Trying to connect to {self.gopro_left_name}...")
            # Connect to LEFT GO PRO
            try:
                pass
                async with WirelessGoPro(target=self.gopro_left_name, enable_wifi=False) as gopro_left:
                    self.logger.info(f"Connected to LEFT GO PRO! ({gopro_left.identifier})")

                    left_task = asyncio.create_task(self.perform_actions(gopro_left,
                                                                        gopro_left_queue))

                    # 1. Retrieve GoPro status for the first time to check everything is ok
                    await gopro_left_queue.put(GoProAction.GET_STATUS_BLE)
                    self.logger.debug(f"Action added to LEFT GO PRO queue: {GoProAction.GET_STATUS_BLE}")
                    
                    # 2. Connect GoPro to the same network as RTMP server
                    await gopro_left_queue.put(GoProAction.CONECT_TO_NETWORK)
                    self.logger.debug(f"Action added to LEFT GO PRO queue: {GoProAction.CONECT_TO_NETWORK}")

                    # 3. Configure streaming
                    await gopro_left_queue.put(GoProAction.CONFIGURE_STREAM)
                    self.logger.debug(f"Action added to LEFT GO PRO queue: {GoProAction.CONFIGURE_STREAM}")

                    # 4. Start streaming
                    await gopro_left_queue.put(GoProAction.START_STREAMING)
                    self.logger.debug(f"Action added to LEFT GO PRO queue: {GoProAction.START_STREAMING}")

                    
                    # Connect to RIGHT GO PRO once connection to LEFT GO PRO has been successful
                    self.logger.info(f"Trying to connect to {self.gopro_right_name}...")
                    async with WirelessGoPro(target=self.gopro_right_name, enable_wifi=False) as gopro_right:
                        self.logger.info(f"Connected to RIGHT GO PRO! ({gopro_right.identifier})")

                        right_task = asyncio.create_task(self.perform_actions(gopro_right, 
                                                                            gopro_right_queue))

                        # 1. Retrieve GoPro status for the first time to check everything is ok
                        await gopro_right_queue.put(GoProAction.GET_STATUS_BLE)
                        self.logger.debug(f"Action added to RIGHT GO PRO queue: {GoProAction.GET_STATUS_BLE}")

                        # 2. Connect GoPro to the same network as RTMP server
                        await gopro_right_queue.put(GoProAction.CONECT_TO_NETWORK)
                        self.logger.debug(f"Action added to RIGHT GO PRO queue: {GoProAction.CONECT_TO_NETWORK}")

                        # 3. Configure streaming
                        await gopro_right_queue.put(GoProAction.CONFIGURE_STREAM)
                        self.logger.debug(f"Action added to RIGHT GO PRO queue: {GoProAction.CONFIGURE_STREAM}")

                        # 4. Start streaming
                        await gopro_right_queue.put(GoProAction.START_STREAMING)
                        self.logger.debug(f"Action added to RIGHT GO PRO queue: {GoProAction.START_STREAMING}")


                        await right_task    # End connection with RIGHT GO PRO
                    self.logger.info(f"Finished connection to RIGHT GO PRO!")

                    await left_task    # End connection with LEFT GO PRO
                    self.logger.info(f"Finished connection to LEFT GO PRO")
                
                await gopro_left_queue.join()
                self.logger.debug(f"All task from LEFT GO PRO Queue done!")
                await gopro_right_queue.join()
                self.logger.debug(f"All task from RIGHT GO PRO Queue done!")
            except FailedToFindDevice as e:
                log_msg = f"Couldn't connect to GoPro, scan timed out without finding a device!"
                self.logger.warning(log_msg)
                self.logger.debug(f"Error message: {e}")
            except OSError as e:
                log_msg = f"OSError!"
                self.logger.error(log_msg)
                self.logger.debug(f"Error message: {e}")

    

    async def start_streaming(self) -> None:
        """
        TBD
        """
        await self.manage_gopros()
