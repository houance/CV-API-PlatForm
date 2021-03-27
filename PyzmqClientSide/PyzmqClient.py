from PyzmqClientSide.CVClient.ResultProcess import *
import requests
import zmq


class PyzmqClient:
    def __init__(self, context, routerUrl, method, confidence=0.8, threshold=0.5):
        self.address = self.GetAddress(routerUrl, method)
        self.context = context
        self.socket = None
        self.socketInit = False
        if method == 'Haar':
            self.clientFrameProcess = CascadesResultProcess.CascadesResultProcess()
        elif method == 'Yolo':
            self.clientFrameProcess = YoloResultProcess.YoloResultProcess(confidence, threshold)
        elif method == 'YuNet':
            self.clientFrameProcess = YuNetResultProcess.YuNetResultProcess(confidence, threshold)

    def predict(self, frame):
        if not self.socketInit:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.address)
            self.socketInit = True
        self.socket.send(self.clientFrameProcess.PyzmqSend(frame))
        return self.clientFrameProcess.ResultProcess(frame, self.socket.recv_pyobj())

    @staticmethod
    def GetAddress(routerUrl, method):
        url = routerUrl + '/' + method
        session = requests.session()
        response = session.post(url)
        return str(response.content, encoding='utf-8')
