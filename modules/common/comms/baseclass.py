"""
Implement base class for communication messages.
"""

from abc import ABC, abstractmethod

from modules.common.comms.definitions import MessageType


__author__ = "EnriqueMoran"


class BaseClass(ABC):
    """
    TBD
    """

    def __init__(self):
        self._message_type = MessageType.UNDEFINED

    
    @abstractmethod
    def pack(self):
        pass


    @abstractmethod
    def unpack(self):
        pass