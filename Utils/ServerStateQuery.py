import psutil
from gpuinfo import GPUInfo
import zmq
import time
import copy


class ServerState:
    def __init__(self, sections: list, addresses: list, cpu=80, gpu=70):
        self.CpUsage = cpu
        self.GpUsage = gpu
        self.sections = sections
        self.addresses = addresses
        self.context = zmq.Context()

    def CpuBusy(self):
        if psutil.cpu_percent(1) > self.CpUsage:
            return False
        else:
            return True

    def GpuBusy(self):
        percent, usage = GPUInfo.gpu_usage()
        if percent[0] > self.GpUsage:
            return False
        else:
            return True

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
