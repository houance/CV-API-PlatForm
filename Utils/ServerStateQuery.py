import psutil
from gpuinfo import GPUInfo
import zmq
import time
import copy


class ServerState:
    def __init__(self, sections: list, addresses: list):
        self.sections = sections
        self.addresses = addresses
        self.context = zmq.Context()

    def CheckConnections(self):
        connectedAddress = copy.deepcopy(self.addresses)
        for index, addresses in enumerate(self.addresses):
            for address in addresses:
                socket = self.context.socket(zmq.REQ)
                socket.connect(address)
                socket.send_string('up')
                time.sleep(2)
                try:
                    recv = socket.recv(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    print('{}:{} server down'.format(self.sections[index], address))
                    connectedAddress[index].remove(address)
                else:
                    print('{}:{} {}'.format(self.sections[index], address, str(recv, encoding='utf-8')))
                socket.disconnect(address)
                socket.close()
        return connectedAddress
