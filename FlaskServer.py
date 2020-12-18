from flask import Flask, request
from YOLOv3 import YOLO
import numpy as np
import cv2
import json


app = Flask(__name__)
yoloModel = YOLO(gpu=1)
cv2.setNumThreads(3)


def decodeJson(jsonData):
    dataDecode = json.loads(jsonData)
    frameDecode = np.array(dataDecode, dtype='uint8')
    frame = cv2.imdecode(frameDecode, cv2.IMREAD_COLOR)
    return frame


@app.route('/yolo', methods=['POST'])
def yoloDetection():
    global yoloModel
    content = request.get_json()
    frame = decodeJson(content['image'])
    frame, boxes = yoloModel.predict(frame)
    flag, frameEncode = cv2.imencode('.jpg', frame, params=[80])
    listEncode = frameEncode.tolist()
    return json.dumps(listEncode)


app.run('127.0.0.1', 5000, debug=True)
