import psutil
from gpuinfo import GPUInfo


class LoadBalance:
    def __init__(self, cpu=80, gpu=70):
        self.CpUsage = cpu
        self.GpUsage = gpu

    def CpuBusy(self):
        if psutil.cpu_percent(1) > self.CpUsage:
            return True
        else:
            return False

    def GpuBusy(self):
        percent, usage = GPUInfo.gpu_usage()
        if percent[0] > self.GpUsage:
            return True
        else:
            return False
