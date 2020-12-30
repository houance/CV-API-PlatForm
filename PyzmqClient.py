import zmq
import cv2


context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5000')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    flag, frameBuffer = cv2.imencode('.jpg', frame, params=[80])
    socket.send(frameBuffer, copy=False)
    socket.recv()

