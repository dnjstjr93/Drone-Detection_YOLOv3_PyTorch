import numpy as np
import os, cv2, glob 
from os import getcwd
from natsort import natsorted
print (os.getcwd())

# wd = "C:/Users/dnjst/Downloads/train_data_0402"
wd = "C:/Users/KETI/Desktop/PyTorch-YOLOv3-master"

files = []
# for file in glob.glob(wd + '/train_data/labels/*.txt'):
for file in glob.glob(wd + '/temp/labels/*.txt'):
	print(file)
	temp = file.split("\\")[-1]
	filename = temp[:-4]
	files.append(filename)

files = natsorted(files)
print(files)
i = 0
while i < len(files):
	file = files[i]
	print(file)
	# image = wd + '/train_data/images/' + file + '.jpg'
	# label = wd + '/train_data/labels/' + file + '.txt'
	image = wd + '/temp/images/' + file + '.jpg'
	label = wd + '/temp/labels/' + file + '.txt'
	print(image)
	with open(label) as f:
			lines = f.readlines()
			for line in lines:
				line = line.split()  #0, cx, cy, w, h
	im = cv2.imread(image)
	# cv2.imshow('im',im)
	#cv2.waitKey(0)
	height, width, channels = im.shape
	#print(line)
	line[1], line[2], line[3], line[4] = float(line[1])*width, float(line[2])*height, float(line[3])*width, float(line[4])*height
	#print(int(float(line[1])-float(line[3])/2.0), int(float(line[2])-float(line[4])/2.0))
	cv2.rectangle(im, (int(line[1]-line[3]/2.0), int(line[2]-line[4]/2.0)), (int(line[1]+line[3]/2.0), int(line[2]+line[4]/2.0)), (0,0,255))
	cv2.imshow('im',im)
	ch = cv2.waitKey(0)
	if ch == 3: i += 1
	elif ch == 2: i -= 1
	elif ch == ord('q'): break