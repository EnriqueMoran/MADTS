"""
TBD
"""

import concurrent.futures
import logging
import os
import socket
import threading

from collections import deque

from modules.VesselDetector.src.baseclass import BaseClass
from modules.VesselDetector.src.configmanager import ConfigManager


__author__ = "EnriqueMoran"


class MulticastManager(BaseClass):
    """
    TBD
    
    Args:
        - config_path (str): Path to configuration file.
        - filename (str): Path to store log file; belongs to BaseClass.
        - format (str): Logger format; belongs to BaseClass.
        - level (str): Logger level; belongs to BaseClass.
    """
    def __init__(self, filename:str, format:str, level:str, config_path:str):
        super().__init__(filename, format, level)
        self.config_parser = ConfigManager(filename=filename, format=format, level=level, 
                                           config_path=config_path)
        self.comm_out = self.config_parser.comm_out
        self.out_sock = None
        self._comms_out_logger = None
        self.set_loggers(filename, format, level)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


    def __del__(self):
        self._stop_communications()

    
    def set_loggers(self, filename, format, level):
        """
        Configure separate loggers for comms_in and comms_out.
        """
        file_name, file_extension = os.path.splitext(filename)
        comms_out_log = file_name + "_comms_out" + file_extension

        self._comms_out_logger = logging.getLogger(self.__class__.__name__ + '_comms_out')
        self._comms_out_logger.setLevel(level)

        if not any(isinstance(h, logging.FileHandler) for h in self._comms_out_logger.handlers):
            comms_out_file_handler = logging.FileHandler(comms_out_log)
            comms_out_file_handler.setLevel(level)
            formatter = logging.Formatter(format)
            comms_out_file_handler.setFormatter(formatter)
            self._comms_out_logger.addHandler(comms_out_file_handler)

        self._comms_out_logger.propagate = False
    
    
    def _create_connections(self):
        """
        TBD
        """        
        self.out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, self.comm_out.ttl)
        self.out_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                 socket.inet_aton(self.comm_out.iface))
        self.logger.debug(f"Output communication sockets created.")


    def send_detection(self, detection):
        """
        TBD
        """
        message = detection.pack()
        binary_data = ''.join(format(byte, '08b') for byte in message)
        hex_data    = ''.join(format(byte, '02x') for byte in message)
        self._comms_out_logger.debug(f"Detection message to send:")
        self._comms_out_logger.debug(f"    Binary data: {binary_data}")
        self._comms_out_logger.debug(f"    Hex data: {hex_data}")
        self._comms_out_logger.debug(f"    x: {detection.x}")
        self._comms_out_logger.debug(f"    y: {detection.y}")
        self._comms_out_logger.debug(f"    width: {detection.width}")
        self._comms_out_logger.debug(f"    height: {detection.height}")
        self._comms_out_logger.debug(f"    probability: {detection.probability}")

        try:
            self.out_sock.sendto(message, (self.comm_out.group, self.comm_out.port))
            msg = f"Detection message sent to: {self.comm_out.group}:{self.comm_out.port}\n"
            self._comms_out_logger.debug(msg)
        except socket.error as e:
            print(f"Error sending Detection message: {e}")
    

    def send_detection_async(self, detection):
        """
        Send detecion asynchronously using a thread from the pool.
        """
        self.executor.submit(self.send_detection, detection)

    
    def start_communications(self):
        """
        Start threads for handling input communications.
        """
        self._create_connections()
        self.logger.debug(f"Output communication initialized...")
    

    def _stop_communications(self):
        """
        Stop the running thread and close sockets.
        """
        if self.out_sock:
            self.out_sock.close()
        self.logger.debug(f"Output communication sockets closed.")