import cv2
import time


class YOLO:
    def __init__(self, gpu=0):
        self.net = cv2.dnn.readNetFromDarknet('/home/nopepsi/PycharmProjects/Vision-System/yolo/yolov3.cfg',
                                              '/home/nopepsi/PycharmProjects/Vision-System/yolo/yolov3.weights')
        if gpu == 1:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, blob):
        self.net.setInput(blob)
        return self.net.forward(self.ln)
