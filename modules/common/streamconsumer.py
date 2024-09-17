"""
Auxiliary class for consuming streaming frame in real time, instead of consuming queued frames.

Based on:
https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
"""

import cv2
import queue
import threading
import time


class StreamConsumer(threading.Thread):

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.queue   = queue.Queue()
        self.stopped = threading.Event()
        self.start()


    def __del__(self):
        self.release()


    def run(self):
        """
        Continuously read frames from the video source, keeping only the most recent one.
        """
        while not self.stopped.is_set() and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()  # Discard the previous frame
                    except queue.Empty:
                        pass

                self.queue.put(frame)

            except cv2.error as e:
                print(f"OpenCV error: {e}")

            except Exception as e:
                print(f"Error in stream reader: {e}")


    def read(self):
        """
        Return the most recent frame from the queue.
        """
        frame = None
        ret   = False
        if not self.queue.empty():
            frame = self.queue.get()
            ret = True
        return ret, frame


    def isOpened(self):
        """
        Check if the video stream is still open.
        """
        return self.cap.isOpened()


    def release(self):
        """
        Stop the stream and release resources.
        """
        self.stopped.set()
        self.join()
        if self.cap:
            self.cap.release()
    
    
    def get_fps(self):
        """
        Return stream fps.
        """
        return self.cap.get(cv2.CAP_PROP_FPS)
