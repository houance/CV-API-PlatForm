from flask import Flask
from Router.RouterUtils.ConfigReader import ConfigReader
from Router.RouterUtils.ServerStateQuery import ServerState

''' 
services >> ['Yolo', 'Haar',......]
addresses >> [[addressOne, addressTwo......], [addressesOne,......]]    
'''
app = Flask(__name__)
reader = ConfigReader('../AddressTable.ini')
services, addressesFromFile = reader.readSectionAndValue()
monitor = ServerState()


@app.route('/' + services[0], methods=['POST'])
def Yolo():
    serverState, address = monitor.ServerState(services[0], addressesFromFile[0])
    if serverState:
        return address
    else:
        return address


@app.route('/' + services[1], methods=['POST'])
def Haar():
    serverState, address = monitor.ServerState(services[1], addressesFromFile[1])
    if serverState:
        return address
    else:
        return address


@app.route('/' + services[2], methods=['POST'])
def YuNet():
    serverState, address = monitor.ServerState(services[2], addressesFromFile[2])
    if serverState:
        return address
    else:
        return address


if __name__ == '__main__':
    app.run('127.0.0.1', 4000, debug=True)
