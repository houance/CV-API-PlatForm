import json
import numpy as np
import requests
import cv2


class PyzmqUtils:
    @staticmethod
    def getAddressAndSendFrameInfo(url, ID, Width, Height):
        header = {'content-type': 'application/json'}
        jsonData = {'ID': ID, 'Width': Width, 'Height': Height}
        response = requests.post(url, data=json.dumps(jsonData), headers=header)
        return response.content

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
    def postJsonRequest(jsonFrame, url, session=None):
        header = {'content-type': 'application/json'}
        jsonRequests = {'image': jsonFrame}
        if not session:
            return requests.post(url, data=json.dumps(jsonRequests), headers=header)
        else:
            return session.post(url, data=json.dumps(jsonRequests), headers=header)
