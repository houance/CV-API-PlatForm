import cv2
from PyzmqServerSide.CVServer.CVServerUtils.NetTransfer import NetTransfer


class YuNet:
    def __init__(self, path='/home/nopepsi/PycharmProjects/CV-API-PlatForm/PyzmqServerSide/CVServer/modelFile/YuNet'
                            '/YuFaceDetectNet_640.onnx'):
        self.detector = cv2.dnn.readNet(path)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outputNames = ['loc', 'conf', 'iou']

    def predict(self, blob):
        computeOutput = []
        self.detector.setInput(blob)
        loc, conf, iou = self.detector.forward(self.outputNames)
        computeOutput.append(loc)
        computeOutput.append(conf)
        computeOutput.append(iou)
        return computeOutput

    def PyzmqPredict(self, recv):
        blob = NetTransfer.decodeYunetBlob(recv)
        return self.predict(blob)
