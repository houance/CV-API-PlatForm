import cv2
import time
from Utils.Utils import jsonUtils
import requests


# cv2.setNumThreads(1)
cap = cv2.VideoCapture(0)
url = 'http://127.0.0.1:5000/yolo'
s = requests.session()
s.keep_alive = True
s.timeout = 1000
counter, sumTime = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    jsonFrame = jsonUtils.packFrame(frame)
    start = time.time()
    response = jsonUtils.postJsonRequest(jsonFrame, url, session=s)
    frame = jsonUtils.decodeJson(response.content)
    end = time.time() - start
    if counter < 5:
        sumTime += end
        counter += 1
    else:
        print(sumTime/10)
        counter = 0
        sumTime = 0
cap.release()
