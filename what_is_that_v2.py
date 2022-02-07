import json
import socket

import cv2
import mediapipe as mp
import numpy as np

from custom_socket import CustomSocket
from hand_tracking_module.hand_tracking import HandTracking
from object_detection_module.object_detection import ObjectDetection

mp_hands = mp.solutions.hands


class WhatIsThat:

    def __init__(self):
        self.OD = ObjectDetection()
        self.HT = HandTracking()

    def what_is_that(self, img):
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # hands detection
        hands_results = self.HT.track(image)
        bbox_list = self.OD.get_bbox(image)
        formatted_bbox = self.OD.format_bbox(bbox_list)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.HT.read_results(image, hands_results)

        # finger_list = [(startindex, midindex, length), ...]
        finger_list = [(7, 8, 200)]

        # define solution list
        obj_list = []

        # check if there is a hand
        if self.HT.hands_results.multi_hand_landmarks:
            self.HT.draw_hand()
            self.HT.draw_hand_label()
            obj_list = self.HT.point_to(formatted_bbox, finger_list)

        self.HT.draw_boxes(formatted_bbox)

        # cv2.imshow('result image', image)
        # cv2.waitKey()
        return obj_list


def main():
    WID = WhatIsThat()

    server = CustomSocket(socket.gethostname(), 10000)
    server.startServer()

    while True:
        conn, addr = server.sock.accept()
        print("Client connected from", addr)
        while True:
            try:
                data = server.recvMsg(conn)
                img = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
                result = WID.what_is_that(img)
                res = {
                    "pointing_at": result
                }
                server.sendMsg(conn, json.dumps(res))
            except Exception as e:
                print(e)
                break


if __name__ == '__main__':
    main()
