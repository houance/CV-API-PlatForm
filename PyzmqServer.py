import zmq
import cv2
from Utils.JsonUtils import jsonUtils
import numpy as np


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5000')
while True:
    frameBuffer = socket.recv()
    frame = cv2.imdecode(np.frombuffer(frameBuffer, dtype='uint8'), -1)
    cv2.imshow('win', frame)
    cv2.waitKey(1)
    socket.send_string('ok')
