import zmq.green as zmq
import cv2
from Utils.threadReadFrame import streamer
from Utils.Utils import PyzmqUtils


class Client:
    def __init__(self, method: str, context):
        self.method = method
        self.streamer = streamer(0)
        self.address = PyzmqUtils.getAddressAndSendFrameInfo('127.0.0.1:8888/' + method, 1, self.streamer.width,
                                                             self.streamer.width)
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.address)

    def framePreProcess(self, frame):
        if self.method == 'Yolo':
            return cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    def sendPreProcesssFrame(self):
        if self.streamer.hasMore():
            frame = self.streamer.getFrame()
        else:
            return
        self.socket.send(self.framePreProcess(frame), copy=False)
        return self.socket.recv()
