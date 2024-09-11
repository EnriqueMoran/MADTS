"""
Auxiliary class for consuming streaming frame in real time, instead of consuming queued frames.

Based on https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
"""

import cv2
import queue
import threading
import time

import cv2
import queue
import threading
import time

class StreamConsumer:

    def __init__(self, url):
        self.url = url
        self.cap = None
        self.queue = queue.Queue()
        self.stopped = False
        self._open()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()


    def __del__(self):
        self.release()


    def _open(self):
        """
        Open the video capture source. Retry until successful.
        """
        while not self.stopped:
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                break
            else:
                self.cap.release()
                time.sleep(1)


    def _reader(self):
        """
        Continuously read frames from the video source, keeping only the most recent one.
        """
        while not self.stopped and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:  # If frame is not successfully read
                    self._reopen()
                    continue

                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()  # Discard the previous frame
                    except queue.Empty:
                        pass

                self.queue.put(frame)

            except cv2.error as e:
                print(f"OpenCV error: {e}")
                self._reopen()

            except Exception as e:
                print(f"Error in stream reader: {e}")
                self._reopen()


    def _reopen(self):
        """
        Attempt to reopen the video stream if an error occurs.
        """
        self.cap.release()
        self._open()


    def read(self):
        """
        Return the most recent frame from the queue.
        """
        frame = None
        ret = False
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
        self.stopped = True
        if self.cap:
            self.cap.release()
