import cv2
import numpy as np


class preProcess:
    @staticmethod
    def encodeFrame(frame, quality=80):
        return cv2.imencode('.jpg', frame, params=[quality])

    @staticmethod
    def decodeFrame(recv):
        return cv2.imdecode(np.frombuffer(recv, dtype='uint8'), -1)

    @staticmethod
    def encodeBlob(frame):
        return cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    @staticmethod
    def decodeBlob(blob):
        return np.frombuffer(blob, dtype='float32').reshape(3, 416, 416)
