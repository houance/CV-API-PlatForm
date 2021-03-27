import requests
import json
import socket


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


class PythonClassSerialize:
    def __init__(self, minGPURequired, maxGpuRequired, clientIP):
        self.minGPURequired = minGPURequired
        self.maxGPURequired = maxGpuRequired
        self.clientIP = clientIP


class KafkaClient:
    def __init__(self, javaCenterIP, minGPURequired, maxGPURequired):
        self.pythonClassSerialize = PythonClassSerialize(minGPURequired, maxGPURequired, get_host_ip())
        self.javaCenterIP = javaCenterIP

    def requesTopicIP(self):
        headers = {'Content-Type': 'application/json'}
        classJson = self.pythonClassSerialize

        response = requests.post(url=self.javaCenterIP, headers=headers,
                                 data=json.dumps(vars(classJson)))
        return response.content
