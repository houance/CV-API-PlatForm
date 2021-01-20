import zmq
from LoadBalance import LoadBalance
from CVServer.CascadesDetector import CascadesDetector
from CVServer.YOLOV3 import YOLO
from CVServer.yuNet import yuNet
from multiprocessing import Process
from threading import Thread


class PyzmqServer:
    def __init__(self, context, address, method):
        self.socket = context.socket(zmq.REP)
        self.socket.bind(address)
        self.method = method
        self.loadBalance = LoadBalance(cpu=80, gpu=70)
        if method == 'Yolo':
            self.cvService = YOLO(gpu=1)
        elif method == 'Haar':
            self.cvService = CascadesDetector()
        elif method == 'YuNet':
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
        if self.method == 'YuNet':
            if self.loadBalance.CpuBusy():
                self.socket.send_string('not ok')
            else:
                self.socket.send_string('ok')
        else:
            if self.loadBalance.GpuBusy():
                self.socket.send_string('not ok')
            else:
                self.socket.send_string('ok')

    def start(self):
        Thread(target=self.service()).start()
