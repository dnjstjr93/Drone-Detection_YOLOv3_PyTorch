# Drone-Detection_YOLOv3_PyTorch

PyTorch based YOLOv3 implementation for drone detection in drone gimbal

reference repo : [https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

## Installation

##### Clone and install requirements

$ git clone https://github.com/dnjstjr93/Drone-Detection_YOLOv3_PyTorch.git

$ cd Drone-Detection_YOLOv3_PyTorch

$ sudo pip3 install -r requirements.txt

##### Download pretrained weights

$ cd weights/

$ bash download_weights.sh

##### Download COCO

$ cd data/

$ bash get_coco_dataset.sh

##### Download Drone and Airplane Dataset

- Download custom dataset such as drone, airplane in [here](https://www.dropbox.com/s/pmid0dy6wgl6c84/data.zip?dl=0)

## Inference

Uses pretrained weights to make predictions on images. Below table displays the inference times when using as inputs
images scaled to 256x256. The ResNet backbone measurements are taken from the YOLOv3 paper. The Darknet-53 measurement
marked shows the inference time of this implementation on my 1080ti card.

$ python3 detect.py --image_folder data/samples/



<p  align="center"><img  src="assets/giraffe.png"  width="480"></p>

<p  align="center"><img  src="assets/dog.png"  width="480"></p>

<p  align="center"><img  src="assets/traffic.png"  width="480"></p>

<p  align="center"><img  src="assets/messi.png"  width="480"></p>

<p  align="center"><img  src="assets/drone_sample_1.png"  width="480"></p>

<p  align="center"><img  src="assets/Video_1_164.png"  width="480"></p>

- __Detect Video__

$ python3 detect_video.py --video_path ./data/video_samples/drone_sample.mp4

<p  align="center"><img  src="assets/drone_sample.gif"  width="480"></p>

## Train

```

$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]

[--gradient_accumulations GRADIENT_ACCUMULATIONS]

[--model_def MODEL_DEF] [--data_config DATA_CONFIG]

[--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]

[--img_size IMG_SIZE]

[--checkpoint_interval CHECKPOINT_INTERVAL]

[--evaluation_interval EVALUATION_INTERVAL]

[--compute_map COMPUTE_MAP]

[--multiscale_training MULTISCALE_TRAINING]

```

#### Example (COCO)

To train on COCO using a Darknet-53 backend pretrained on ImageNet run:

```

$ python3 train.py --data_config config/coco.data --pretrained_weights weights/darknet53.conv.74

```

#### Training log

```

---- [Epoch 7/100, Batch 7300/14658] ----

+------------+--------------+--------------+--------------+

| Metrics | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |

+------------+--------------+--------------+--------------+

| grid_size | 16 | 32 | 64 |

| loss | 1.554926 | 1.446884 | 1.427585 |

| x | 0.028157 | 0.044483 | 0.051159 |

| y | 0.040524 | 0.035687 | 0.046307 |

| w | 0.078980 | 0.066310 | 0.027984 |

| h | 0.133414 | 0.094540 | 0.037121 |

| conf | 1.234448 | 1.165665 | 1.223495 |

| cls | 0.039402 | 0.040198 | 0.041520 |

| cls_acc | 44.44% | 43.59% | 32.50% |

| recall50 | 0.361111 | 0.384615 | 0.300000 |

| recall75 | 0.222222 | 0.282051 | 0.300000 |

| precision | 0.520000 | 0.300000 | 0.070175 |

| conf_obj | 0.599058 | 0.622685 | 0.651472 |

| conf_noobj | 0.003778 | 0.004039 | 0.004044 |

+------------+--------------+--------------+--------------+

Total Loss 4.429395

---- ETA 0:35:48.821929

```

## Train on Custom Dataset

#### Custom model

Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your
dataset.

```

$ cd config/ # Navigate to config dir

$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'

```

#### Classes

Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder

Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder

Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the
image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should
define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be
scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name
in `data/custom/classes.names`.

#### Define Train and Validation Sets

In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation
data respectively.

#### Train

To train on the custom dataset run:

```

$ python3 train-drone.py --model_def config/yolov3-custom.cfg --data_config config/custom.data

```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.
