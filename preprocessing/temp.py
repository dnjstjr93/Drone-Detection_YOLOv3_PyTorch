import cv2, os, shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import argparse


# path = "C:/Users/dnjst/Downloads/zcsj2g2m4c-4/Database1/"
# image_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/images/"
# label_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/labels/"

# # for filename in os.listdir(path):
# #     name = filename.split(".")[0]
# #     if (filename.split(".")[-1] == 'txt'):
# #         label = open(path+filename, 'r')
# #         line = label.readline()
# #         if (line == ''):
# #             label.close()
# #         else:
# #             print(name)
# #             img = cv2.imread(path+name+'.JPEG')
# #             cv2.imwrite(image_path+'{}.jpg'.format(name), img)
# #             shutil.copy(path+filename, label_path+filename)


# train_images_path = "C:/Users/dnjst/Downloads/train_data_0402/train_data/images/"
# train_labels_path = "C:/Users/dnjst/Downloads/train_data_0402/train_data/labels/"
# for filename in os.listdir(train_labels_path):
#     name = filename.split(".")[0]
#     label = open(train_labels_path+filename, 'r')
#     line = label.readline()
#     if (line == ''):
#         label.close()
#     else:
#         try:
#             print(name)
#             shutil.copy(train_images_path+name+".jpg", image_path+name+".jpg")
#             shutil.copy(train_labels_path+filename, label_path+filename)
#         except cv2.error:
#             pass
