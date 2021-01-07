import cv2
import numpy as np
from Utils.Nms import nms
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
        idx = np.where(dets[:, -1] > self.confidence)[0]
        dets = dets[idx]

        if dets.shape[0]:
            facess = nms(dets, self.threshold)
        else:
            facess = ()
        faces = np.array(facess[:, :4])
        faces = faces.astype(np.int)
        faceStartXY = faces[:, :2]
        faceEndXY = faces[:, 2:4]
        faceWH = faceEndXY - faceStartXY
        faces = np.hstack((faceStartXY, faceWH))
        for (x, y, w, h) in faces:
            cv2.rectangle(frameNew, (x, y), (x + w, y + h), (0, 0, 255))
        return frameNew, faces

    @staticmethod
    def CascadesResultProcess(frame, faces: np.ndarray):
        frameNew = frame.copy()
        if faces is None:
            return False, None
        faces = faces.flatten().reshape((-1, 4))
        for face in faces:
            cv2.rectangle(frameNew, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255))
        return frameNew, faces

    def YOLOResultProcess(self, frame, LayerOutputs, detectionFilter=-1, painted=1):
        frameNew = frame.copy()
        frameNewHeight, frameNewWidth = frameNew.shape[:2]
        layerOutputs = LayerOutputs

        boxes = []
        confidences = []
        classIDs = []
        detectionBoxes = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if isinstance(detectionFilter, list):
                    filterList = set(detectionFilter)
                    if classID + 1 not in filterList:
                        continue

                elif detectionFilter != -1:
                    if classID != detectionFilter - 1:
                        continue

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the frameNew, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([frameNewWidth, frameNewHeight, frameNewWidth, frameNewHeight])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.colours[classIDs[i]]]
                text = "{}: {:.4f}".format(self.labels[int(classIDs[i])], confidences[i])
                if painted:
                    cv2.rectangle(frameNew, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frameNew, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detectionBoxes.append([x, y, w, h, classIDs[i]])
        return frameNew, detectionBoxes
