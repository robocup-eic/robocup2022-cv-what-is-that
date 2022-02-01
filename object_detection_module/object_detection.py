
import time
import cv2
import torch
# import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from .utils.datasets import letterbox
from .utils.general import non_max_suppression, scale_coords, xyxy2xywh
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, time_synchronized

from .models.models import *
import os

# path

CONFIG_PATH = 'object_detection_module/config/'
WEIGHTS_PATH = CONFIG_PATH + 'yolor_p6.pt'
NAMES_PATH = CONFIG_PATH + 'coco.names'
DEVICE = "cpu"
CFG_PATH = CONFIG_PATH + 'yolor_p6.cfg'
IMAGE_SIZE = 1280


class ObjectDetection:

    def __init__(self):
        self.device = select_device(DEVICE)
        # half precision only supported on CUDA
        self.half = self.device.type != 'cpu'

        # load model
        # .cuda() #if you want cuda remove the comment
        self.model = Darknet(CFG_PATH, IMAGE_SIZE)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device)['model'])
        self.model.to(self.device).eval()

        if self.half:
            self.model.half()

        # Get names and colors
        self.names = self.load_classes(NAMES_PATH)
        self.color = [255, 0, 0]

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        # filter removes empty strings (such as last line)
        return list(filter(None, names))

    def detect(self, input_image):

        # preprocess image
        input_image = self.preprocess(input_image)

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)  # init img
        # run once
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        # Padded resize
        img = letterbox(input_image, new_shape=IMAGE_SIZE, auto_size=64)[0]

        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        print("recieving image with shape {}".format(img.shape))

        img = torch.from_numpy(img).to(self.device)
        # uint8 to fp16/32
        img = img.half() if self.half else img.float()
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        print("Inferencing ...")
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False)

        print("found {} object".format(len(pred)))

        # print string
        s = ""
        s += '%gx%g ' % img.shape[2:]

        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], input_image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:

                    # python yolor_example.py
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, input_image, label=label,
                                 color=self.color, line_thickness=3)

        # Print time (inference + NMS)q
        print('{}Done. {:.3} s'.format(s, time.time() - t0))

        return input_image

    def get_bbox(self, input_image):

        #preprocess image
        input_image = self.preprocess(input_image)

        # object bbox list
        bbox_list = []

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)  # init img
        # run once
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        # Padded resize
        img = letterbox(input_image, new_shape=IMAGE_SIZE, auto_size=64)[0]

        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        print("recieving image with shape {}".format(img.shape))

        img = torch.from_numpy(img).to(self.device)
        # uint8 to fp16/32
        img = img.half() if self.half else img.float()
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        print("Inferencing ...")
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False)

        print("found {} object".format(len(pred)))

        # print string
        s = ""
        s += '%gx%g ' % img.shape[2:]

        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], input_image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    temp = []
                    for ts in xyxy:
                        temp.append(ts.item())
                    bbox = list(np.array(temp).astype(int))
                    bbox.append(self.names[int(cls)])
                    bbox_list.append(bbox)

        # Print time (inference + NMS)q
        print('{}Done. {:.3} s'.format(s, time.time() - t0))

        return bbox_list

    # format bbox list for mediapipe
    def format_bbox(self, bbox_list):
        format_bboxs = []
        for bbox in bbox_list:
            if bbox[4] == 'person':
                pass
            else:
                format_bboxs.append([bbox[4], tuple([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]), False])
        return format_bboxs

    def preprocess(self, img):
        npimg = np.array(img)
        image = npimg.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def main():
    # create model
    OD = ObjectDetection()

    # load our input image and grab its spatial dimensions
    img = cv2.imread("./test1.jpg")

    # choose one method
    with torch.no_grad():
        # get detected image
        res = OD.detect(img)

        # get bboxs of object in images
        bboxs = OD.get_bbox(img)

    # show output
    image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.imshow('yolor', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
