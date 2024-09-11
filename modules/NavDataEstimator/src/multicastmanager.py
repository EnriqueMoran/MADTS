"""
TBD
"""

import concurrent.futures
import logging
import os
import socket
import threading

from collections import deque

from modules.common.comms import baseclass, definitions
from modules.common.comms.detection import Detection
from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager


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
        self.comm_in  = self.config_parser.comm_in
        self.comm_out = self.config_parser.comm_out
        self.in_sock  = None
        self.out_sock = None
        self.running  = False    # Is receive data thread running?
        self.input_thread = None
        self._comms_in_logger  = None
        self._comms_out_logger = None
        self.set_loggers(filename, format, level)
        self.detection_buffer = deque(maxlen=self.config_parser.stream.max_detections)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


    def __del__(self):
        self._stop_communications()

    
    def set_loggers(self, filename, format, level):
        """
        Configure separate loggers for comms_in and comms_out.
        """
        file_name, file_extension = os.path.splitext(filename)
        comms_in_log  = file_name + "_comms_in" + file_extension
        comms_out_log = file_name + "_comms_out" + file_extension

        self._comms_in_logger = logging.getLogger(self.__class__.__name__ + '_comms_in')
        self._comms_in_logger.setLevel(level)

        if not any(isinstance(h, logging.FileHandler) for h in self._comms_in_logger.handlers):
            comms_in_file_handler = logging.FileHandler(comms_in_log)
            comms_in_file_handler.setLevel(level)
            formatter = logging.Formatter(format)
            comms_in_file_handler.setFormatter(formatter)
            self._comms_in_logger.addHandler(comms_in_file_handler)

        self._comms_in_logger.propagate = False

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
        self.in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, self.comm_in.ttl)
        self.in_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, 
                                socket.inet_aton(self.comm_in.iface))
        self.in_sock.bind((self.comm_in.iface, self.comm_in.port))

        mreq = socket.inet_aton(self.comm_in.group) + socket.inet_aton(self.comm_in.iface)
        self.in_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        self.out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, self.comm_out.ttl)
        self.out_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, 
                                 socket.inet_aton(self.comm_out.iface))
        self.logger.debug(f"Input and output communication sockets created.")
            

    def send_nav_data(self, nav_data):
        """
        TBD
        """
        message = nav_data.pack()
        binary_data = ''.join(format(byte, '08b') for byte in message)
        hex_data    = ''.join(format(byte, '02x') for byte in message)
        self._comms_out_logger.debug(f"NavData message to send:")
        self._comms_out_logger.debug(f"    Binary data: {binary_data}")
        self._comms_out_logger.debug(f"    Hex data: {hex_data}")
        self._comms_out_logger.debug(f"    id: {nav_data.id}")
        self._comms_out_logger.debug(f"    distance: {nav_data.distance}")
        self._comms_out_logger.debug(f"    bearing: {nav_data.bearing}")

        try:
            self.out_sock.sendto(message, (self.comm_out.group, self.comm_out.port))
            msg = f"NavData message sent to: {self.comm_out.group}:{self.comm_out.port}\n"
            self._comms_out_logger.debug(msg)
        except socket.error as e:
            print(f"Error sending NavData message: {e}")
    

    def send_nav_data_async(self, nav_data):
        """
        Send navigation data asynchronously using a thread from the pool.
        """
        self.executor.submit(self.send_nav_data, nav_data)


    def get_message(self, message):
        """
        TBD
        """
        message_type = baseclass.BaseClass.get_type(message)
        res = None

        self._comms_in_logger.debug(f"Message type: {message_type}")
        if message_type == definitions.MessageType.DETECTION:
            res = self.get_detection(message)
        else:
            self._comms_in_logger.debug(f"Message not processed due to unrecognized type!")
        
        return (message_type, res)


    def get_detection(self, detection_msg):
        """
        TBD
        """
        unpacked_data = Detection.unpack(detection_msg)
        pos_x  = unpacked_data['x']
        pos_y  = unpacked_data['y']
        width  = unpacked_data['width']
        height = unpacked_data['height']
        probability = unpacked_data['probability']

        self._comms_in_logger.debug(f"Detection message unpacked:")
        self._comms_in_logger.debug(f"    x: {pos_x}")
        self._comms_in_logger.debug(f"    y: {pos_y}")
        self._comms_in_logger.debug(f"    width: {width}")
        self._comms_in_logger.debug(f"    height: {height}")
        self._comms_in_logger.debug(f"    probability: {probability}")
        return unpacked_data
        
    
    def start_communications(self):
        """
        Start threads for handling input communications.
        """
        self._create_connections()
        self.running = True
        self.input_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.input_thread.start()
        self.logger.debug(f"Input communication thread running...")
    

    def _stop_communications(self):
        """
        Stop the running thread and close sockets.
        """
        self.running = False
        if self.input_thread:
            self.input_thread.join()
        self.logger.debug(f"Input communication thread stopped.")

        if self.in_sock:
            self.in_sock.close()
        if self.out_sock:
            self.out_sock.close()
        self.logger.debug(f"Input and output communication sockets closed.")


    def receive_loop(self):
        """
        Loop to handle receiving messages.
        """
        while self.running:
            try:
                message, addr = self.in_sock.recvfrom(1024)
                self._comms_in_logger.debug(f"Received message from {addr}: {message}")
                message_type, data = self.get_message(message)
                if message_type == definitions.MessageType.DETECTION:
                    self._comms_in_logger.debug(f"Processed detection message: {data}")

                    detection = Detection()
                    detection.x = data['x']
                    detection.y = data['y']
                    detection.width  = data['width']
                    detection.height = data['height']
                    detection.probability = data['probability']

                    self.detection_buffer.append(detection)
                    msg = f"Detection added to buffer. Len: {len(self.detection_buffer)}"
                    self._comms_in_logger.debug(msg)
                    msg = f"Detection buffer: {self.detection_buffer}"
                    self._comms_in_logger.debug(msg)
            except socket.error as e:
                self._comms_in_logger.error(f"Socket error: {str(e)}")