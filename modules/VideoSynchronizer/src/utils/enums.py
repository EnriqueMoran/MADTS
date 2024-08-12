"""
TBD
"""

from enum import Enum


class GoProAction(Enum):
    """
    TBD
    """
    GET_STATUS_BLE       = 1
    CONFIGURE_STREAM     = 2
    CONECT_TO_NETWORK    = 3
    CHECK_GOPRO_IS_READY = 4
    START_STREAMING      = 5
    STOP_STREAMING       = 6

