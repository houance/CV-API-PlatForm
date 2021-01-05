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
rett, frameEncode = NetTransfer.encodeFrame(frame, flag=1)
socket.send(frameEncode, copy=False)
recv = socket.recv_pyobj()
frame1, detection = ResultProcess.CascadesResultProcess(frame, recv)
cap.release()

