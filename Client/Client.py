import zmq.green as zmq
from threading import Thread
from Utils.threadReadFrame import streamer
from Utils.PyzmqUtils import PyzmqUtils
from Utils.PreProcessUtils import preProcess
from queue import Queue


class Client:
    def __init__(self, method: str, context, queueSize=128):
        self.method = method
        self.streamer = streamer(0)
        self.address = PyzmqUtils.getAddressAndSendFrameInfo('127.0.0.1:8888/' + method, 1, self.streamer.width,
                                                             self.streamer.width)
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.address)
        self.queue = Queue(maxsize=queueSize)
        self.startThread()

    def framePreProcess(self, frame):
        if self.method == 'Yolo':
            return preProcess.YoloPreProcess(frame)

    def sendAndRecvFrame(self):
        if self.streamer.hasMore():
            frame = self.streamer.getFrame()
        else:
            return False
        self.socket.send(self.framePreProcess(frame), copy=False)
        return self.socket.recv()

    def startThread(self):
        t = Thread(target=self.sendAndRecvFrame(), args=())
        t.daemon = True
        t.start()
        return self
