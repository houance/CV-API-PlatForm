import cv2
import numpy as np


class NetTransfer:
    @staticmethod
    def encodeFrame(frame, quality=80, flag=1):
        if flag == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.imencode('.jpg', frame, params=[quality])

    @staticmethod
    def decodeFrame(recv):
        return cv2.imdecode(np.frombuffer(recv, dtype='uint8'), -1)

    @staticmethod
    def encodeBlob(frame):
        return cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    @staticmethod
    def decodeYoloBlob(blob):
        return np.frombuffer(blob, dtype='float32').reshape(1, 3, 416, 416)

    @staticmethod
    def decodeDetection(detection):
        return np.frombuffer(detection, dtype='int').reshape(-1, 5)
