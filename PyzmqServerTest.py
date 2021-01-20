import zmq
from PyzmqServer.PyzmqServer import PyzmqServer
from multiprocessing import Process

context = zmq.Context()
server = PyzmqServer(context, 'tcp://127.0.0.1:8000', 'YuNet')
server.start()
