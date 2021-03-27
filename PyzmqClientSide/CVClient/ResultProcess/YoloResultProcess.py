import cv2
import numpy as np
from PyzmqClientSide.CVClient.CVClientUtils.NetTransfer import NetTransfer


class YoloResultProcess:
    def __init__(self, confidence=0.8, threshold=0.5):
        np.random.seed(42)
        self.labels = open('/home/nopepsi/PycharmProjects/Vision-System/yolo/coco.names').read().strip().split('\n')
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.confidence = confidence
        self.threshold = threshold

    def ResultProcess(self, frame, layerOutputs, detectionFilter=-1, painted=1):


        frameNew = frame.copy()
        frameNewHeight, frameNewWidth = frameNew.shape[:2]
        detection = np.concatenate((layerOutputs[0], layerOutputs[1], layerOutputs[2]))
        boxes = detection[:, :4] * np.array([frameNewWidth, frameNewHeight, frameNewWidth, frameNewHeight])
        boxes = list(boxes.astype('int'))
        classIDS = np.argmax(detection[:, 5:], axis=1)
        confidence = np.amax(detection[:, 5:], axis=1)
        boxesIndex = cv2.dnn.NMSBoxes(boxes, confidence, self.confidence, self.threshold)
        detections = []

        if len(boxesIndex) > 0:
            for i in boxesIndex.flatten():
                x = boxes[i][0] = int(boxes[i][0] - boxes[i][2] / 2)
                y = boxes[i][1] = int(boxes[i][1] - boxes[i][3] / 2)
                detections.append(np.append(boxes[i], classIDS[i]))
                color = [int(c) for c in self.colours[classIDS[i]]]
                text = "{}: {:.4f}".format(self.labels[int(classIDS[i])], confidence[i])
                w = boxes[i][2]
                h = boxes[i][3]
                if painted:
                    cv2.rectangle(frameNew, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frameNew, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frameNew, detections

    @staticmethod
    def PyzmqSend(frame):
        return NetTransfer.encodeYoloBlob(frame)
