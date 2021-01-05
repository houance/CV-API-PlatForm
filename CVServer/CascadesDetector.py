import cv2
import numpy as np


class CascadesDetector:
    def __init__(self):
        self.detector = cv2.cuda.CascadeClassifier_create('/home/nopepsi/PycharmProjects/Vision-System/faceDetect'
                                                          '/haarcascade_frontalface_default_cuda.xml')

    def predict(self, frame):
        gpuFrame = cv2.cuda_GpuMat()
        gpuFrame.upload(frame)
        gpuGrayFrame = cv2.cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY)
        gpuResult = self.detector.detectMultiScale(gpuGrayFrame)
        return gpuResult.download()
