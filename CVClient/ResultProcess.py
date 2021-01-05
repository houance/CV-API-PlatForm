import cv2
import numpy as np


class ResultProcess:
    def __init__(self, confidence=0.5, threshold=0.3):
        np.random.seed(42)
        self.labels = open('/home/nopepsi/PycharmProjects/Vision-System/yolo/coco.names').read().strip().split('\n')
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.confidence = confidence
        self.threshold = threshold

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
