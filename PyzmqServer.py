import zmq
from Utils.PreProcessUtils import preProcess

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5000')
while True:
    blob = preProcess.decodeBlob(socket.recv())
    socket.send(blob, copy=False)
