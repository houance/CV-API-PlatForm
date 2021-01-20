import zmq
import time
import copy


class ServerState:
    def __init__(self):
        self.context = zmq.Context()

    def CheckConnections(self, section, addresses):
        connectedAddress = copy.deepcopy(addresses)
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            socket.send_string('up', copy=False)
            time.sleep(1)
            try:
                recv = socket.recv(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                print('{}:{} server down'.format(section, address))
                connectedAddress.remove(address)
            else:
                print('{}:{} {}'.format(section, address, str(recv, encoding='utf-8')))
            socket.disconnect(address)
            socket.close()
        return connectedAddress

    def ServerState(self, section, targetAddresses):
        connectedAddresses = self.CheckConnections(section, targetAddresses)
        if not connectedAddresses:
            return False, 'All {} Server Down'.format(section)
        for address in connectedAddresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            socket.send_string('okk', copy=False)
            recv = socket.recv_string()
            socket.disconnect(address)
            socket.close()
            if recv == 'ok':
                return True, address
            elif recv == 'not ok':
                return False, 'No Server Usable'

