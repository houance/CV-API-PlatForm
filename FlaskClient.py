import cv2
import time
from Utils.JsonUtils import jsonUtils


cv2.setNumThreads(1)
cap = cv2.VideoCapture(0)
url = 'http://127.0.0.1:5000/yolo'
while True:
    ret, frame = cap.read()
    if not ret:
        break
    jsonFrame = jsonUtils.packFrame(frame)
    response = jsonUtils.postJsonRequest(jsonFrame, url)
    frame = jsonUtils.decodeJson(response.content)
    cv2.imshow('win', frame)
    cv2.waitKey(1)
cap.release()
