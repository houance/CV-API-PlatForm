from PyzmqServer import PyzmqServer
import zmq

context = zmq.Context()
# service1 = PyzmqServer(context, 'tcp://127.0.0.1:6000', 'Yolo')
# service2 = PyzmqServer(context, 'tcp://127.0.0.1:7000', 'Haar')
service3 = PyzmqServer(context, 'tcp://127.0.0.1:8000', 'YuNet')
# for index, section in enumerate(sections):
#     for address in addresses[index]:
#         service = PyzmqServer(context, address, section)
#         service.start()
