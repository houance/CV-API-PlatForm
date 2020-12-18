import requests
import cv2
import json
import numpy as np
import time


def packAndPost(url, frame):
    flag, frameEncode = cv2.imencode('.jpg', frame, params=[80])
    listEncode = frameEncode.tolist()
    data = {'image': '{}'.format(json.dumps(listEncode))}
    header = {'content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=header)
    return response


def decodeJson(jsonData):
    dataDecode = json.loads(jsonData)
    frameDecode = np.array(dataDecode, dtype='uint8')
    frame = cv2.imdecode(frameDecode, cv2.IMREAD_COLOR)
    return frame


cv2.setNumThreads(1)
cap = cv2.VideoCapture(0)
while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    response = packAndPost('http://127.0.0.1:5000/yolo', frame1)
    start = time.time()
    frame = decodeJson(response.content)
    print(time.time() - start)
    cv2.imshow('win', frame)
    cv2.waitKey(1)
cap.release()
