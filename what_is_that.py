from object_detection_module.object_detection import ObjectDetection
from hand_tracking_module.hand_tracking import HandTracking
import torch
import numpy as np
import mediapipe as mp
import cv2

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

        cv2.imshow('result image', image)
        cv2.waitKey()


def main():
    # init model
    WID = WhatIsThat()

    # load img
    img_test = cv2.imread('test_pics/test_hand.jpg')

    # feed img to model
    i_see = WID.what_is_that(img_test)

    print(i_see)

    # # test
    # while True:
    #     a = int(input())
    #     if a == 0:
    #         break
    #     img_path = 'test_pics/test' + str(a) + '.jpg'
    #     img_test = cv2.imread(img_path)
    #     print(img_test.shape)
    #     i_see = WID.what_is_that(img_test)





if __name__ == '__main__':
    main()
