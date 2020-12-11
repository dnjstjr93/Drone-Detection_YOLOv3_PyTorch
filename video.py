from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import time
import argparse
import cv2
import subprocess

cap = cv2.VideoCapture(0)
NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)

rtmp_url = "rtmp://203.253.128.135:1935/live01/drone01"

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

command = "appsrc ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc bitrate=4000000 ! video/x-h264,stream-format=(string)byte-stream,alignment=(string)au ! h264parse ! queue !  flvmux name=mux ! rtmpsink location={}".format(rtmp_url)

output = cv2.VideoWriter(command, 0, fps, (width, height))

count = 0
while True:
    ret, img = cap.read()
    if ret is False:
        break
    cv2.imshow('Detector', img)

                # p.stdin.write(img.tobytes())
    output.write(img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
