import requests
from flask import Flask, request
import json
from threading import Thread
from time import sleep
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
    def __init__(self, serverIP, job):
        self.serverIP = serverIP
        self.job = job
        self.alive = True


class KafkaServer:
    def __init__(self, javaCenterIP, job):
        self.javaCenterIP = javaCenterIP
        self.pythonServerSerialize = PythonClassSerialize('http://127.0.0.1:4000', job)
        self.headers = {'Content-Type': 'application/json'}
        self.thread = Thread(target=self.heartBeat)

    def heartBeat(self):
        group = False
        while True:
            if not group:
                response = requests.post(url=self.javaCenterIP + '/group', headers=self.headers,
                                         data=json.dumps(vars(self.pythonServerSerialize)))
                print(str(response.content, 'utf8'))
                if str(response.content, 'utf8') == 'group':
                    group = True
            else:
                requests.post(url=self.javaCenterIP + '/heartbeat', headers=self.headers,
                              data=json.dumps(vars(self.pythonServerSerialize)))

            sleep(5)

    def startHeartBeat(self):
        self.thread.start()


if __name__ == '__main__':
    app = Flask(__name__)
    kafkaServer = KafkaServer('http://127.0.0.1:8080/PythonServer', 'Yolo')
    kafkaServer.startHeartBeat()


    @app.route('/ReceiveTopicPartition', methods=['POST'])
    def receiveTopicPartition():
        return 'a'
