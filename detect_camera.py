# -*-coding:utf-8 -*-

"""
 Created by Wonseok Jung in KETI on 2021-06-07.
"""

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import time
import argparse
import cv2
import subprocess

import torch
from torch.autograd import Variable

def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img


point_x = 0
point_y = 0


def move_event(event, x, y, flags, params):
    global width
    global height
    global point_x
    global point_y

    if event == cv2.EVENT_MOUSEMOVE:
        point_x = x
        point_y = y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--video_file", type=str, default=0, help="path to dataset")
    # parser.add_argument("--video_file", type=str, default=0, help="path to dataset")
    # parser.add_argument("--video_file", type=str, default="rtmp://203.253.128.135:1935/live01/drone01", help="path to dataset")
    # parser.add_argument("--video_file", type=str, default="./data/video_samples/drone_sample.mp4", help="path to dataset")
    # parser.add_argument("--video_file", type=str, default="./data/video_samples/17_12-00-00.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/yolov3-drone.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_499.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/coco.names", help="path to class label file")
    # parser.add_argument("--class_path", type=str, default="data/custom/classes-drone.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #if opt.video_file.endswith(".mp4"):
    cap = cv2.VideoCapture(opt.video_file)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    time_begin = time.time()
    NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #NUM=0

    rtmp_url = "rtmp://203.253.128.135:1935/live01/drone01"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    while cv2.waitKey(1) < 0:
        ret, img = cap.read()
        if ret:
            # img = cv2.resize(img, None, fx=0.4, fy=0.4)

            RGBimg=changeBGR2RGB(img)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)

            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(Tensor))

            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            a.clear()
            if detections is not None:
                a.extend(detections)
            b=len(a)
            if len(a):
                for detections in a:
                    if detections is not None:
                        detections = rescale_boxes(detections, opt.img_size, RGBimg.shape[:2])
                        unique_labels = detections[:, -1].cpu().unique()
                        n_cls_preds = len(unique_labels)
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            color = [int(c) for c in colors[int(cls_pred)]]
                            img = cv2.rectangle(img, (int(x1), int(y1 + box_h)), (int(x2), int(y1)), color, 2)
                            cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(img, str("%.2f" % float(conf)), (int(x2), int(y2 - box_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color, 2)

            cv2.setMouseCallback('Detector', move_event)

            result = changeRGB2BGR(img)
            cv2.putText(result, 'x={}, y={}'.format(point_x, point_y), (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 255, 255), 2)
            cv2.imshow('Detector', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('ERRRRRRRRRRRROR')

    time_end = time.time()
    time_total = time_end - time_begin
    print(time_total)

cap.release()
cv2.destroyAllWindows()
