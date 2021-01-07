import zmq
import cv2
from Utils.NetTransfer import NetTransfer
import numpy as np
import time
from CVClient.ResultProcess import ResultProcess
from Utils.threadReadFrame import streamer

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5000')
visualize = ResultProcess(threshold=0.7)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
blobEncode = NetTransfer.encodeYunetBlob(frame)
socket.send(blobEncode, copy=False)
recv = socket.recv_pyobj()
frame1, detection = visualize.yuNet(frame, recv)
cv2.imshow('win', frame1)
cv2.waitKey(100)
cv2.destroyWindow('win')
cap.release()

