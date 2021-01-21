from PyzmqServer import PyzmqServer
import zmq


context = zmq.Context()
service1 = PyzmqServer(context, 'tcp://127.0.0.1:6000', 'Yolo')
service2 = PyzmqServer(context, 'tcp://127.0.0.1:7000', 'Haar')
service3 = PyzmqServer(context, 'tcp://127.0.0.1:8000', 'YuNet')

# p1 = Process(target=service1.service, args=(context,))
# p2 = Process(target=service2.service, args=(context,))
# p3 = Process(target=service3.service, args=(context,))
# print(1)
# p1.start()
# print(2)
# p2.start()
# print(3)
# p3.start()
# print(4)
# for index, section in enumerate(sections):
#     for address in addresses[index]:
#         service = PyzmqServer(context, address, section)
#         service.start()
