import cv2


class yuNet:
    def __init__(self, threshold=0.3, path='modelFile/yuNet/YuFaceDetectNet_640.onnx'):
        self.detector = cv2.dnn.readNet(path)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.threshold = threshold

    def predict(self, blob):
        computeOutput = []
        outputNames = ['loc', 'conf', 'iou']
        self.detector.setInput(blob)
        loc, conf, iou = self.detector.forward(outputNames)
        computeOutput.append(loc)
        computeOutput.append(conf)
        computeOutput.append(iou)
        return computeOutput
