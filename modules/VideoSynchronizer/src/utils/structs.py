from dataclasses import dataclass

__author__ = "EnriqueMoran"


@dataclass
class RTMP:
    left_camera_url:str
    right_camera_url:str
    network_ssid:str
    network_password:str


@dataclass
class GoPro:
    left_camera_name:str
    right_camera_name:str


@dataclass
class Stream:
    resolution:int
    min_bitrate:int
    max_bitrate:int
    starting_bitrate:int
    fov:int
    fps:int
    duration:int
    hypersmooth:bool
    record_stream:bool