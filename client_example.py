import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

img = cv2.imread("test_pics/test1.jpg")
print(img.shape)
img = cv2.resize(img, (480, 640))

host = socket.gethostname()
port = 10000

c = CustomSocket(host,port)
c.clientConnect()
result = c.req(img)
print(result)
print(result['pointing_at'])


