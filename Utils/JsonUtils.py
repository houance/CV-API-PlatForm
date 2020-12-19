import json
import numpy as np
import requests
import cv2


class jsonUtils:
    @staticmethod
    def packFrame(frame, method='jpg', quality=80):
        flag, frameEncode = cv2.imencode('.{}'.format(method), frame, params=[quality])
        listEncode = frameEncode.tolist()
        return json.dumps(listEncode)

    @staticmethod
    def decodeJson(jsonData):
        dataDecode = json.loads(jsonData)
        frameDecode = np.array(dataDecode, dtype='uint8')
        return cv2.imdecode(frameDecode, cv2.IMREAD_COLOR)

    @staticmethod
    def postJsonRequest(jsonFrame, url):
        header = {'content-type': 'application/json'}
        jsonRequests = {'image': jsonFrame}
        return requests.post(url, data=json.dumps(jsonRequests), headers=header)
