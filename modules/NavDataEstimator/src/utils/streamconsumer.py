"""
Auxiliary class for consuming streaming frame in real time, instead of consuming queued frames.

Based on https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
"""

import cv2
import queue
import threading
import time


class StreamConsumer:

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.queue = queue.Queue()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    
    def __del__(self):
        self.cap.release()
    

    def _reader(self):
        """
        Read frames as soon as they are available, keeping only most recent one.
        """
        while self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
            except Exception as e:    # Stream ended
                print(e)
                break
            if not ret:
                break
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.queue.put(frame) 
    
    def read(self):
        frame = None
        ret   = False
        if not self.queue.empty():
            frame = self.queue.get()
            ret   = True
        return ret, frame
    

    def isOpened(self):
        return self.cap.isOpened()


    def release(self):
        self.cap.release()