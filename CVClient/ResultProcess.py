import cv2
import numpy as np


class ResultProcess:
    @staticmethod
    def ObjectVisulize(frame, detections):
        for detection in detections:
            x = detection[0]
            y = detection[1]
            w = detection[2]
            h = detection[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
