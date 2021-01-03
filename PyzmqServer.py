import zmq
from Utils.PreProcessUtils import preProcess
from CVServer.YOLOV3 import YOLO

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5000')
yolo = YOLO(gpu=1)
while True:
    blob = preProcess.decodeBlob(socket.recv())

    socket.send(yolo.predict(blob), copy=False)
