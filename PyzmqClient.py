import zmq
import cv2
from Utils.NetTransfer import NetTransfer
from CVClient.ResultProcess import ResultProcess

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5000')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    blob = NetTransfer.encodeBlob(frame)
    socket.send(blob)
    ResultProcess.ObjectVisulize(frame, NetTransfer.decodeDetection(socket.recv()))
    cv2.imshow('win', frame)
    cv2.waitKey(1)

# ret, frame = cap.read()
# blob = preProcess.encodeBlob(frame)
# socket.send(blob)
# recv = preProcess.decodeDetection(socket.recv())
# YOLOV3Client.visualize(frame, recv)
# cv2.imshow('win', frame)
# cv2.waitKey(0)
