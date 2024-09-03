"""
TBD
"""

from dataclasses import dataclass


__author__ = "EnriqueMoran"


@dataclass
class RTMP:
    left_camera_url: str = None
    right_camera_url:str = None
    network_ssid:    str = None
    network_password:str = None


@dataclass
class GoPro:
    left_camera_name: str = None
    right_camera_name:str = None


@dataclass
class Stream:
    resolution: int = None
    min_bitrate:int = None
    max_bitrate:int = None
    starting_bitrate:int = None
    fov:int = None
    fps:int = None
    duration:int = None
    hypersmooth:  bool = None
    record_stream:bool = None