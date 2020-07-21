import cv2, os, shutil
import pandas as pd

# Set PATH
ori_path = "C:/Users/dnjst/Desktop/drone-detection-master/DataSets/Drones/positive/"
dir_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/"
os.makedirs("temp", exist_ok=True)
data_path = "C:/Users/dnjst/Desktop/PyTorch-YOLOv3-master/temp/"

for dataset in os.listdir(ori_path):
    # Check the folder
    if not os.path.exists(data_path+'images'):
        os.makedirs(data_path+'images')
    if not os.path.exists(data_path+'bbox'):
        os.makedirs(data_path+'bbox')
    if not os.path.exists(data_path+'labels'):
        os.makedirs(data_path+'labels')
    
    # Data read & parsing
    filename = dataset.split('.')[0]
    extension = dataset.split('.')[-1]

    # If the file extension is tsv,
    if (extension == 'tsv'):
        filename_1 = dataset.split('.')[-2]
        if (filename_1 == 'labels'):
            label = pd.read_csv(ori_path+dataset, delimiter='\t', header=0)
            label = list(label)
            if (label[0] == 'drone'):
                res_label = 0
        elif (filename_1 == 'bboxes'):
            bbox = pd.read_csv(ori_path+dataset, delimiter='\t', header=0)
            bbox = list(bbox)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            width = x2 - x1
            height = y2 - y1

            # Create the text files include label, bounding box data
            txt = open(data_path+'bbox/'+(dataset.split('.')[0])+".txt", 'w')
            data = '{} {} {} {} {}'.format(res_label, x1, y1, width, height)
            txt.write(data)
            txt.close()

            # Here, scales and translates bounding box data
            img = cv2.imread(ori_path+filename+'.jpg')
            h, w, ch = img.shape

            scaled_width = width / w
            scaled_height = height / h
            x_center = ((x1 / w) + (scaled_width / 2))
            y_center = ((y1 / h) + (scaled_height / 2))
            calcurated_bbox = [x_center, y_center, scaled_width, scaled_height]

            labels_txt = open(data_path + 'labels/' + (dataset.split('.')[0]) + ".txt", 'w')
            scaled_data = '{} {} {} {} {}'.format(res_label, calcurated_bbox[0], calcurated_bbox[1],
                                                  calcurated_bbox[2], calcurated_bbox[3])
            labels_txt.write(scaled_data)
            labels_txt.close()

    # If the file extension is JSON,
    # TBD #    

    # Copy the image file
    elif (extension == 'jpg'):
        shutil.copy(ori_path+dataset, data_path+'images/'+dataset)