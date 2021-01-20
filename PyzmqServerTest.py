import zmq
from Server.PyzmqServer import PyzmqServer
from multiprocessing import Process

context = zmq.Context()
server = PyzmqServer(context, 'tcp://127.0.0.1:5000', 'yuNet')
server2 = PyzmqServer(context, 'tcp://127.0.0.1:6000', 'haar')
server2.start()
server.start()
