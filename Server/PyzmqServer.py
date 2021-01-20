import zmq
from Server.LoadBalance import LoadBalance
from CVServer.CascadesDetector import CascadesDetector
from CVServer.YOLOV3 import YOLO
from CVServer.yuNet import yuNet
from multiprocessing import Process


class PyzmqServer:
    def __init__(self, context, address, method):
        self.socket = context.socket(zmq.REP)
        self.socket.bind(address)
        self.method = method
        self.loadBalance = LoadBalance(cpu=80, gpu=70)
        if method == 'yolo':
            self.cvService = YOLO(gpu=1)
        elif method == 'haar':
            self.cvService = CascadesDetector()
        elif method == 'yuNet':
            self.cvService = yuNet()
        self.start()

    def service(self):
        while True:
            recv = self.socket.recv(copy=False)
            if len(recv) == 2:
                self.AnswerServerUp()
            elif len(recv) == 3:
                self.LoadBalance()
            else:
                self.socket.send_pyobj(self.cvService.PyzmqPredict(recv), copy=False)

    def AnswerServerUp(self):
        self.socket.send_string('server up')

    def LoadBalance(self):
        if self.method == 'yuNet':
            if self.loadBalance.CpuBusy():
                self.socket.send_string('not ok')
        else:
            if self.loadBalance.GpuBusy():
                self.socket.send_string('not ok')

    def start(self):
        Process(target=self.service()).start()
