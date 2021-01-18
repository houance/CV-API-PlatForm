from flask import Flask, jsonify
from Utils.ConfigReader import ConfigReader
from Utils.ServerStateQuery import ServerState


''' 
services >> ['Yolo', 'Haar',......]
addresses >> [[addressOne, addressTwo......], [addressesOne,......]]    
'''
app = Flask(__name__)
reader = ConfigReader('Router/AddressTable.ini')
services, addresses = reader.readSectionAndValue()


@app.route(services[0], methods=['POST'])
def yoloAddress():
    return 'hello'


if __name__ == '__main__':
    app.run('127.0.0.1', 5000, debug=True)
