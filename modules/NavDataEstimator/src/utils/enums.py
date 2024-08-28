"""
TBD
"""

from enum import Enum


__author__ = "EnriqueMoran"


class UndistortMethod(Enum):
    """
    TBD
    """
    UNDISTORT = 1
    REMAP     = 2


class CalibrationMode(Enum):
    """
    TBD
    """
    USE_VIDEOS = 1
    USE_IMAGES = 2


class RectificationMode(Enum):
    """
    TBD
    """
    CALIBRATED_SYSTEM   = 1
    UNCALIBRATED_SYSTEM = 2
    MATCHING_KEYPOINTS  = 3