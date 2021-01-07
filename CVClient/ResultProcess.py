import cv2
import numpy as np
from Utils.PriorBox import PriorBox


class ResultProcess:
    def __init__(self, confidence=0.5, threshold=0.3):
        np.random.seed(42)
        self.labels = open('/home/nopepsi/PycharmProjects/Vision-System/yolo/coco.names').read().strip().split('\n')
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.confidence = confidence
        self.threshold = threshold
        self.init = False
        self.pb = None

    def yuNet(self, frame, result):
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
    def turnIntoRectBoxes(xyBoxes):
        startXY = xyBoxes[:, :2]
        WH = xyBoxes[:, 2:4] - startXY
        return np.hstack((startXY, WH))

    @staticmethod
    def CascadesResultProcess(frame, faces: np.ndarray):
        frameNew = frame.copy()
        if faces is None:
            return False, None
        faces = faces.flatten().reshape((-1, 4))
        for face in faces:
            cv2.rectangle(frameNew, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255))
        return frameNew, faces

    def YOLOResultProcess(self, frame, layerOutputs, detectionFilter=-1, painted=1):
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
