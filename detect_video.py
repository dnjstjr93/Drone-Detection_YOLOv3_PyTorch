from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./data/video_samples/DJI_sample.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="./config/yolov3-drone.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_499.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="./data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    # parser.add_argument("--checkpoint_model", type=str, default="checkpoints/yolov3_ckpt_499.pth", help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    ### Read video & Save frame
    video_path = opt.video_path
    print("\nvideo_path: ", video_path)
    frame_name = video_path.split("/")[3].split(".")[0]
    print("frame_name: ", frame_name)
    save_frame_folder = video_path.split("/")[2]
    print("save_frame_folder: ", save_frame_folder)
    frame_folder = "data/{}/{}".format(save_frame_folder, frame_name)
    print("frame_folder: ", frame_folder)
    result_frame_folder = "output/{}".format(frame_name)
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(result_frame_folder, exist_ok=True)

    cam  = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    num_of_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("num_of_frame: ", num_of_frame)

    count = 0
    while (cam.isOpened()):
        ret, frame = cam.read()

        if (count < (num_of_frame-1)):
            # ret, frame = cam.read()
            cv2.imwrite(frame_folder + "/{}_{}.jpg".format(frame_name,count), frame)
        else:
            break

        count += 1

    dataloader = DataLoader(
        ImageFolder(frame_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0] # Linux
        filename = path.split("\\")[-1].split(".")[0] # Windows
        plt.savefig("{}/{}.png".format(result_frame_folder, filename), bbox_inches="tight", pad_inches=0.0)
        # plt.close()
        plt.clf()

    ### Save result frame as video
    os.makedirs("output/Video", exist_ok=True)

    result_frame_array = []
    for i in range(num_of_frame-1):
        result_frame = result_frame_folder + "/" + frame_name + "_{}.png".format(i)
        # reading each files
        img = cv2.imread(result_frame)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        result_frame_array.append(img)

    out = cv2.VideoWriter("output/Video/{}.mp4".format(frame_name), cv2.VideoWriter_fourcc(*'MP4V'), 20.0, size)

    for i in range(len(result_frame_array)):
        # writing to a image array
        out.write(result_frame_array[i])
    out.release()