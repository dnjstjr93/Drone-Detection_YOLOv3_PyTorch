import os, shutil

# wd = "C:/Users/dnjst/Downloads/train_data_0402"
wd = "C:/Users/KETI/Desktop/PyTorch-YOLOv3-master"
coco_path = "E:/data/coco"

# Change drone, airplane label => sum with coco dataset
count = 0
for file in os.listdir(wd + '/data/custom/labels/'):
	if file[:-4].isdigit():
	# airplane
		if ((int(file[:-4]) > 34308) and (int(file[:-4]) < 2266585)):
			with open(wd + "/data/custom/labels/" + file) as f:
				lines = f.readlines()
				for line in lines:
					line = line.split()  # 0, cx, cy, w, h
					line[0] = 81
					data = '{} {} {} {} {}'.format(line[0], line[1], line[2], line[3], line[4])
			txt = open(wd + "/data/custom/labels/" + file, 'w')
			txt.write(data)
			txt.close()

	# drone
	else:
		with open(wd + "/data/custom/labels/" + file) as f:
			lines = f.readlines()
			if len(lines) > 1:
				d1 = lines[0].split()
				d2 = lines[1].split()
				d1[0] = 80
				d2[0] = 80
				data = '{} {} {} {} {}\n{} {} {} {} {}'.format(d1[0], d1[1], d1[2], d1[3], d1[4], d2[0], d2[1], d2[2], d2[3], d2[4])

			else:
				for line in lines:
					line = line.split()  # 0, cx, cy, w, h
					line[0] = 80
					data = '{} {} {} {} {}'.format(line[0], line[1], line[2], line[3], line[4])
		txt = open(wd + "/data/custom/labels/" + file, 'w')
		txt.write(data)
		txt.close()

# # Copy & Rename coco
# # images/train2017
# for file in os.listdir(coco_path + '/images/train2017'):
# 	shutil.copy(coco_path + '/images/train2017/' + file, wd + '/data/custom/images/coco_'+file)
# # images/val2017
# for file in os.listdir(coco_path + '/images/val2017'):
# 	shutil.copy(coco_path + '/images/val2017/' + file, wd + '/data/custom/images/coco_'+file)
# # labels/train2017
# for file in os.listdir(coco_path + '/labels/train2017'):
# 	shutil.copy(coco_path + '/labels/train2017/' + file, wd + '/data/custom/labels/coco_'+file)
# # labels/val2017
# for file in os.listdir(coco_path + '/labels/val2017'):
# 	shutil.copy(coco_path + '/labels/val2017/' + file, wd + '/data/custom/labels/coco_'+file)
#
# # train2017.txt
# shutil.copy(coco_path + '/train2017.txt', wd + '/data/custom/coco_train2017.txt')
# # change label path
# with open(wd + "/data/custom/coco_train2017.txt") as f:
# 	lines = f.readlines()
# 	t = [0 for i in range(len(lines))]
# 	for i in range(len(lines)):
# 		filename = lines[i].split('/')[3]
# 		t[i] = lines[i]
# 		t[i] = 'data/custom/images/coco_' + filename
#
# txt = open(wd + '/data/custom/coco_train2017.txt', 'w')
# for w in range(len(t)):
# 	txt.write(t[w])
# txt.close()
# # val2017.txt
# shutil.copy(coco_path + '/val2017.txt', wd + '/data/custom/coco_val2017.txt')
# # change label path
# with open(wd + "/data/custom/coco_val2017.txt") as f:
# 	lines = f.readlines()
# 	v = [0 for i in range(len(lines))]
# 	for i in range(len(lines)):
# 		filename = lines[i].split('/')[3]
# 		v[i] = lines[i]
# 		v[i] = 'data/custom/images/coco_' + filename
#
# txt = open(wd + '/data/custom/coco_val2017.txt', 'w')
# for w in range(len(v)):
# 	txt.write(v[w])
# txt.close()
#
# # coco.names & coco2017.data
# shutil.copy(coco_path + '/../coco.names', wd + '/data/custom/coco.names')
# shutil.copy(coco_path + '/../coco2017.data', wd + '/config/coco2017.data')
