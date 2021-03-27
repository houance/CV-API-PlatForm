import cv2
import numpy as np
from PyzmqClientSide.CVClient.CVClientUtils.PriorBox import PriorBox
from PyzmqClientSide.CVClient.CVClientUtils.NetTransfer import NetTransfer


class YuNetResultProcess:
    def __init__(self, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.init = False
        self.pb = None

    def ResultProcess(self, frame, result):
        if not self.init:
            frameWidth, frameHeight = frame.shape[:2]
            self.pb = PriorBox(input_shape=(640, 480), output_shape=(frameHeight, frameWidth))
            self.init = True

        frameNew = frame.copy()
        loc = result[0]
        conf = result[1]
        iou = result[2]
        dets = self.pb.decode(np.squeeze(loc, axis=0), np.squeeze(conf, axis=0), np.squeeze(iou, axis=0))

        XY = dets[:, :4].astype('int').reshape(-1, 4)
        boxes = list(self.turnIntoRectBoxes(XY))
        confidences = list(dets[:, -1].astype('float'))
        idx = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        faces = []
        if len(idx) > 0:
            for i in idx.flatten():
                faces.append(boxes[i])
                (x, y, w, h) = boxes[i]
                cv2.rectangle(frameNew, (x, y), (x + w, y + h), (0, 0, 255))

        return frameNew, faces

    @staticmethod
    def PyzmqSend(frame):
        return NetTransfer.encodeYunetBlob(frame)

    @staticmethod
    def turnIntoRectBoxes(xyBoxes):
        startXY = xyBoxes[:, :2]
        WH = xyBoxes[:, 2:4] - startXY
        return np.hstack((startXY, WH))
