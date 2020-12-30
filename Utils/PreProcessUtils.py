import cv2


class preProcess:
    @staticmethod
    def YoloPreProcess(frame):
        return cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

