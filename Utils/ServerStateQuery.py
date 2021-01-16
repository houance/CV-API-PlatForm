import psutil
from gpuinfo import GPUInfo


class ServerState:
    def __init__(self, cpu=80, gpu=70):
        self.CpUsage = cpu
        self.GpUsage = gpu

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
