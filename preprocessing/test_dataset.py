import cv2, os, shutil, random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import argparse

# path = "C:/Users/dnjst/Desktop/drone-detection-master/DataSets/Drones/positive/"
# data_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/"
# for dataset in os.listdir(path):
#     # if i < 10:
#     if ((dataset.split('.')[-1]) == 'tsv'):
#         if ((dataset.split('.')[-2]) == 'labels'):
#             print((dataset.split('.')[-2]))
#             label = pd.read_csv(path+dataset, delimiter='\t', header=0)
#             label = list(label)
#             if (label[0] == 'drone'):
#                 res_label = 0
#         elif ((dataset.split('.')[-2]) == 'bboxes'):
#             print((dataset.split('.')[-2]))
#             bbox = pd.read_csv(path+dataset, delimiter='\t', header=0)
#             bbox = list(bbox)
#             x_center = int(bbox[0])
#             y_center = int(bbox[1])
#             x_end_center = int(bbox[2])
#             y_end_center = int(bbox[3])
#             width = int(bbox[2]) - x_center
#             height = int(bbox[3]) - y_center
#     elif ((dataset.split('.')[-1]) == 'jpg'):
#         shutil.copy(path+dataset, data_path+'images/'+dataset)
#         txt = open(data_path+'labels/'+(dataset.split('.')[0])+".txt", 'w')
#         data = '{} {} {} {} {}'.format(res_label, x_center, y_center, width, height)
#         txt.write(data)

train_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/images/"
label_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/labels/"
image_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/image/"
img_path = "C:/Users/dnjst/Downloads/zcsj2g2m4c-4/Database1/"
train_txt = open("C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/train.txt", 'w')
valid_txt = open("C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/valid.txt", 'w')
i = 0

files = os.listdir(train_path)
index = random.randrange(0, len(files))
for image in range(len(files)):
    index = random.randrange(0, len(files))
    if (i < 20234):
        train_txt.write('data/custom/images/'+files[index]+'\n')
    else:
        train_txt.close()
        valid_txt.write('data/custom/images/'+files[index]+'\n')
    i += 1
valid_txt.close()
print(i)

read_train_txt = open("C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/train.txt", 'r')
read_valid_txt = open("C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/data/custom/valid.txt", 'r')
i = 0
while True:
    line = read_train_txt.readline()
    if not line: break
    i += 1
print(i)
read_train_txt.close() 

i = 0
while True:
    line = read_valid_txt.readline()
    if not line: break
    i += 1
print(i)
read_valid_txt.close()

# for label_txt in os.listdir(label_path):
#     print(label_txt)
#     label = open(label_path+label_txt, 'r')
#     line = label.readline()
#     if (line == ''):
#         print("++++++++++++++++++++++++",label_txt)
#         label.close()
#         os.remove(label_path+label_txt)
# for label_txt in os.listdir(label_path):
#     name = label_txt.split(".")[0] + ".jpg"
#     for image in os.listdir(img_path):
#         # print(label_txt.split(".")[0])
#         # print(image.split(".")[0])
#         if (image.split(".")[0] == label_txt.split(".")[0]):
#             shutil.copy(img_path+image, train_path+name)

# for image in os.listdir(train_path):
#     for img in os.listdir(img_path):
#     # print(label_txt.split(".")[0])
#     # print(image.split(".")[0])
#         if (image.split(".")[0] == img.split(".")[0]):
#             shutil.copy(img_path+img, image_path+img)

# for label_txt in os.listdir(label_path):
#     label = open(label_path+label_txt, 'r')
#     name = label_txt.split('.')[0]
#     line = label.readline()
#     data = line.split(' ')
#     annotation = int(data[0])
#     x = int(data[1])
#     y = int(data[2])
#     x_end = int(data[3])
#     y_end = int(data[4])
#     print(x)
#     print(y)
#     print(x_end)
#     print(y_end)
#     # for file in range(1, 351):
#     #     if (name == '{}'.format(file)):
#     #         img = cv2.imread(train_path+name+'.jpg')
#     #         cv2.rectangle(img, (x, y), (x_end, y_end), (0,0,255), 2)
#     img = cv2.imread(train_path+name+'.jpg')
#     img = cv2.rectangle(img, (x, y), (x+x_end, y+y_end), (0,0,255), 2)
#     cv2.imwrite(image_path+'{}.jpg'.format(name), img)

#     label.close()


'''
parser = argparse.ArgumentParser()
parser.add_argument("--data_config", type=str, default="config/custom-drone.data", help="path to data config file")
opt = parser.parse_args()

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    print(path)
    fp = open(path, "r")
    print(fp.read())
    names = fp.read().split("\n")[:-1]
    return names

data_config = parse_data_config(opt.data_config)
print(data_config['names'])
class_names = load_classes(data_config["names"])
print(class_names)
'''
'''
DB_image_path = "C:/Users/dnjst/Downloads/zcsj2g2m4c-4/Database1/images/"
DB_label_path = "C:/Users/dnjst/Downloads/zcsj2g2m4c-4/Database1/labels/"
DB_result_path = "C:/Users/dnjst/Downloads/zcsj2g2m4c-4/Database1/result/"
i = 0
for image_txt in os.listdir(DB_image_path):
    if i < 10:
        name = image_txt.split('.')[0]
        img = cv2.imread(DB_image_path+image_txt)
        label = open(DB_label_path+name+'.txt', 'r')
        line = label.readline()
        if (line != ''):
            data = line.split(' ')
            annotation = int(data[0])
            x = int(float(data[1])*100)
            y = int(float(data[2])*100)
            x_end = int(float(data[3])*100)
            y_end = int(float(data[4])*100)
            print(x)
            print(y)
            print(x_end)
            print(y_end)
            # for file in range(1, 351):
            #     if (name == '{}'.format(file)):
            #         img = cv2.imread(train_path+name+'.jpg')
            #         cv2.rectangle(img, (x, y), (x_end, y_end), (0,0,255), 2)
            img = cv2.imread(DB_image_path+image_txt)
            img = cv2.rectangle(img, (x, y), (x+x_end, y+y_end), (0,0,255), 2)
            cv2.imwrite(DB_result_path+'{}.jpg'.format(name), img)

            label.close()
    i += 1
'''