import zmq
import cv2
from Utils.PreProcessUtils import preProcess

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5000')
cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     blob = preProcess.encodeBlob(frame)
#     socket.send(blob)
#     socket.recv()
ret, frame = cap.read()
blob = preProcess.encodeBlob(frame)
socket.send(blob)
blobRecv = preProcess.decodeBlob(socket.recv())
cap.release()
