from threading import Thread
import cv2
from queue import Queue
import time


class streamer:
    def __init__(self, path, queueSize=128):
        self.cap = cv2.VideoCapture(path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.queue = Queue(maxsize=queueSize)
        self.stop = False
        self.startThread()
        time.sleep(1)

    def startThread(self):
        t = Thread(target=self.readFrame, args=())
        t.daemon = True
        t.start()
        return self

    def readFrame(self):
        while True:
            if self.stop:
                return

            if not self.queue.full():
                ret, frame = self.cap.read()

                if not ret:
                    self.stop = True
                    self.cap.release()
                    break
                self.queue.put(frame)
            else:
                continue

    def hasMore(self):
        if not self.queue.empty():
            return True
        else:
            return False

    def getFrame(self):
        return self.queue.get()
