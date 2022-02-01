import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

img = cv2.imread("test_pics/test3.jpg")
print(img.shape)

host = socket.gethostname()
port = 10000

c = CustomSocket(host,port)
c.clientConnect()
result = c.whatIsThat(img)
print(result)
print(result['pointing_at'])


