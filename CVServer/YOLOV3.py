import cv2
import numpy as np
import time


class YOLO:
    def __init__(self, gpu=0, confidence=0.5, threshold=0.3):
        self.net = None
        self.ln = None
        self.yolov3Init(gpu)
        self.confidence = confidence
        self.threshold = threshold

    def yolov3Init(self, gpu):
        self.net = cv2.dnn.readNetFromDarknet('/home/nopepsi/PycharmProjects/Vision-System/yolo/yolov3.cfg',
                                              '/home/nopepsi/PycharmProjects/Vision-System/yolo/yolov3.weights')
        if gpu == 1:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, blob, detectionFilter=-1):
        imageNewWidth = 640
        imageNewHeight = 480
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

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
                    # the size of the imageNew, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([imageNewWidth, imageNewHeight, imageNewWidth, imageNewHeight])
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
                detectionBoxes.append([x, y, w, h, classIDs[i]])
        return np.array(detectionBoxes)
