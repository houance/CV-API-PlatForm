import zmq
from Utils.NetTransfer import NetTransfer
from CVServer.YOLOV3 import YOLO
from CVServer.CascadesDetector import CascadesDetector
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5000')
yolo = YOLO(gpu=1)
faceDetector = CascadesDetector()
while True:
    frame = NetTransfer.decodeFrame(socket.recv())
    result = faceDetector.predict(frame)

    print(result)
    socket.send_pyobj(result, copy=False)
