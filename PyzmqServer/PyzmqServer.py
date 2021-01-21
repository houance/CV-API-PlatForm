import zmq.green as zmq
from LoadBalance import LoadBalance
from CVServer.CascadesDetector import CascadesDetector
from CVServer.YOLOV3 import YOLO
from CVServer.yuNet import yuNet
from multiprocessing import Process
import time



class PyzmqServer:
    def __init__(self, context, address, method):
        self.address = address
        self.method = method
        self.socket = None
        self.loadBalance = LoadBalance(cpu=80, gpu=70)
        if method == 'Yolo':
            self.cvService = YOLO(gpu=1)
        elif method == 'Haar':
            self.cvService = CascadesDetector()
        elif method == 'YuNet':
            self.cvService = yuNet()
        self.startProcess(context)

    def service(self, context):
        self.socket = context.socket(zmq.REP)
        self.socket.bind(self.address)
        while True:
            time.sleep(0.1)
            recv = self.socket.recv(copy=False)
            if len(recv) == 2:
                self.answerServerUp()
            elif len(recv) == 3:
                self.answerServerBusy()
            else:
                self.socket.send_pyobj(self.cvService.PyzmqPredict(recv), copy=False)

    def answerServerUp(self):
        self.socket.send_string('server up')

    def answerServerBusy(self):
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

    def startProcess(self, context):
        Process(target=self.service, args=(context,)).start()
