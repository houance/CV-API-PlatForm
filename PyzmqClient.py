import zmq
import cv2
from Utils.NetTransfer import NetTransfer
import time
from CVClient.ResultProcess import ResultProcess

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5000')
visualize = ResultProcess(threshold=0.2, confidence=0.5)
cap = cv2.VideoCapture('/home/nopepsi/PycharmProjects/Vision-System/output1.avi')
while True:
    ret, frame = cap.read()
    if not ret:
        cv2.destroyWindow('win')
        break
    blobEncode = NetTransfer.encodeYoloBlob(frame)
    socket.send(blobEncode, copy=False)
    recv = socket.recv_pyobj()
    frame1, detection = visualize.YOLOResultProcess(frame, recv)
    cv2.imshow('win',  frame1)
    cv2.waitKey(1)

