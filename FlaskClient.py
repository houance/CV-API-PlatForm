import cv2
import time
from Utils.JsonUtils import jsonUtils
import requests


cv2.setNumThreads(1)
cap = cv2.VideoCapture(0)
url = 'http://127.0.0.1:5000/yolo'
s = requests.session()
s.keep_alive = True
s.timeout = 1000
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    jsonFrame = jsonUtils.packFrame(frame)
    response = jsonUtils.postJsonRequest(jsonFrame, url, session=s)
    frame = jsonUtils.decodeJson(response.content)
    print(time.time() - start)
    cv2.imshow('win', frame)
    cv2.waitKey(1)
cap.release()
