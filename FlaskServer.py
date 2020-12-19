from flask import Flask, request
import numpy as np
import cv2
from Utils.JsonUtils import jsonUtils

app = Flask(__name__)
cv2.setNumThreads(3)


@app.route('/yolo', methods=['POST'])
def yoloDetection():
    content = request.get_json()
    frame = jsonUtils.decodeJson(content['image'])
    return jsonUtils.packFrame(frame)


if __name__ == '__main__':
    app.run('127.0.0.1', 5000, threaded=True, debug=True)
