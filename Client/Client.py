import zmq.green as zmq
from Utils.threadReadFrame import streamer
from Utils.PyzmqUtils import PyzmqUtils
from Utils.PreProcessUtils import preProcess


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
            return preProcess.encodeBlob(frame)

    def sendAndRecvFrame(self):
        if self.streamer.hasMore():
            frame = self.streamer.getFrame()
        else:
            return False
        self.socket.send(self.framePreProcess(frame), copy=False)
        return preProcess.decodeFrame(self.socket.recv())

