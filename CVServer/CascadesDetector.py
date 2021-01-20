import cv2
from Utils.NetTransfer import NetTransfer
import zmq


class CascadesDetector:
    def __init__(self, path='modelFile/haar/haarcascade_frontalface_default_cuda.xml'):
        self.detector = cv2.cuda.CascadeClassifier_create(path)

    def predict(self, frame):
        gpuFrame = cv2.cuda_GpuMat()
        gpuFrame.upload(frame)
        gpuGrayFrame = cv2.cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY)
        gpuResult = self.detector.detectMultiScale(gpuGrayFrame)
        return gpuResult.download()

    def PyzmqPredict(self, recv):
        frame = NetTransfer.decodeFrame(recv)
        return self.predict(frame)
