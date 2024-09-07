"""
Implement base class for communication messages.
"""

import struct

from abc import ABC, abstractmethod

from modules.common.comms.definitions import MessageType


__author__ = "EnriqueMoran"


class BaseClass(ABC):
    """
    TBD
    """

    def __init__(self):
        self._message_type = MessageType.UNDEFINED


    @staticmethod
    def get_type(message):
        """
        TBD
        """
        return MessageType(struct.unpack('>B', message[0:1])[0])

    
    @abstractmethod
    def pack(self):
        pass


    @abstractmethod
    def unpack(self):
        pass