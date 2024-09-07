"""
TBD
"""

import socket
import threading

from modules.common.comms import baseclass, definitions, detection, navdata
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
        self.running  = False    # Threads running?
        self.in_thread  = None
        self.out_thread = None
        self._start_communications()


    def __del__(self):
        self._stop_communications()
        
    
    def _create_connections(self):
        """
        TBD
        """
        self.in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, self.comm_in.ttl)
        self.in_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, 
                                socket.inet_aton(self.comm_in.iface))
        
        self.out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
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
        self.logger.debug(f"NavData message to send:")
        self.logger.debug(f"    Binary data: {binary_data}")
        self.logger.debug(f"    Hex data: {hex_data}")
        self.logger.debug(f"    id: {nav_data.id}")
        self.logger.debug(f"    distance: {nav_data.distance}")
        self.logger.debug(f"    bearing: {nav_data.bearing}")

        try:
            self.in_sock.sendto(message, (self.comm_in.group, self.comm_in.port))
            self.logger.debug(f"NavData message sent to: {self.comm_in.group}:{self.comm_in.port}")
        except socket.error as e:
            print(f"Error sending NavData message: {e}")
    

    def get_message(self, message):
        """
        TBD
        """
        message_type = baseclass.BaseClass.get_type(message)
        res = None

        if message_type == definitions.MessageType.DETECTION:
            res = self.get_detection(message)
        
        return (message_type, res)


    def get_detection(self, detection_msg):
        """
        TBD
        """
        unpacked_data = detection.Detection.unpack(detection_msg)
        pos_x  = unpacked_data['x']
        pos_y  = unpacked_data['y']
        width  = unpacked_data['width']
        height = unpacked_data['height']
        probability = unpacked_data['probability']

        self.logger.debug(f"Detection message unpacked:")
        self.logger.debug(f"    x: {pos_x}")
        self.logger.debug(f"    y: {pos_y}")
        self.logger.debug(f"    width: {width}")
        self.logger.debug(f"    height: {height}")
        self.logger.debug(f"    probability: {probability}")
        return unpacked_data
        
    
    def _start_communications(self):
        """
        Start threads for handling input and output communications.
        """
        self._create_connections()
        self.running = True
        self.input_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.output_thread = threading.Thread(target=self.send_loop, daemon=True)

        self.input_thread.start()
        #self.output_thread.start()
        self.logger.debug(f"Input and output communication threads running...")
    

    def _stop_communications(self):
        """
        Stop the running threads and close sockets.
        """
        self.running = False
        if self.input_thread:
            self.input_thread.join()
        if self.output_thread:
            self.output_thread.join()
        self.logger.debug(f"Input and output communication threads stopped.")

        if self.in_sock:
            self.in_sock.close()
        if self.out_sock:
            self.out_sock.close()
        self.logger.debug(f"Input and output communication sockets closed.")


    def send_loop(self):
        """
        Loop to handle sending messages.
        """
        while self.running:
            nav_data = navdata.NavData()    # TODO
            self.send_nav_data(nav_data)
            threading.Event().wait(1 / self.comm_out.frequency)


    def receive_loop(self):
        """
        Loop to handle receiving messages.
        """
        while self.running:
            try:
                message, addr = self.in_sock.recvfrom(1024)
                self.logger.debug(f"Received message from {addr}: {message}")
                message_type, data = self.get_message(message)
                if message_type == definitions.MessageType.DETECTION:
                    self.logger.debug(f"Processed detection message: {data}")
            except socket.error as e:
                self.logger.error(f"Socket error: {e}")