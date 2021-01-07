import zmq
from Utils.NetTransfer import NetTransfer
from CVServer.YOLOV3 import YOLO
from CVServer.CascadesDetector import CascadesDetector
from CVServer.yuNet import yuNet


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5000')
yolo = YOLO(gpu=1)
faceDetector = CascadesDetector()
faceDetector2 = yuNet()
while True:
    blob = NetTransfer.decodeYunetBlob(socket.recv())
    result = faceDetector2.predict(blob)
    socket.send_pyobj(result, copy=False)
