import cv2
import numpy as np
from PyzmqClientSide.CVClient.CVClientUtils.NetTransfer import NetTransfer


class CascadesResultProcess:
    @staticmethod
    def ResultProcess(frame, faces: np.ndarray):
        frameNew = frame.copy()
        if faces is None:
            return False, None
        faces = faces.flatten().reshape((-1, 4))
        for face in faces:
            cv2.rectangle(frameNew, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255))
        return frameNew, faces

    @staticmethod
    def PyzmqSend(frame):
        return NetTransfer.encodeFrame(frame)
