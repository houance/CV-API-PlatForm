import zmq.green as zmq
from PyzmqServerSide.CVServer.CVPredict import *
from PyzmqServerSide.CVServer.CVServerUtils.LoadBalance import LoadBalance
from multiprocessing import Process
import time


class PyzmqServer:
    def __init__(self, context, routerAddress, method):
        self.address = routerAddress
        self.method = method
        self.socket = None
        self.loadBalance = LoadBalance(cpu=80, gpu=70)
        if method == 'Yolo':
            self.cvService = YOLOV3.YOLO(gpu=1)
        elif method == 'Haar':
            self.cvService = CascadesDetector.CascadesDetector()
        elif method == 'YuNet':
            self.cvService = YuNet.YuNet()
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

    @staticmethod
    def StartServers(context, sections, addresses):
        for index, section in enumerate(sections):
            for address in addresses[index]:
                PyzmqServer(context, address, section)
