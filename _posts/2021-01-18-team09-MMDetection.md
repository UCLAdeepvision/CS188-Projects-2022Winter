---
layout: post
comments: true
title: MMDetection
author: Yu Zhou, Zongyang Yue
date: 2022-01-27
---
# Introduction

> In this paper, we study state of the art object detection algorithms and their implementations in MMDetection. We measure their performances on main-stream benchmarks such as the COCO dataset, and further evaluate their performances against adversarial attacks.
We explore and try to understand the library by removing parts of it and checking the change in performances of our algorithms.


<!--more-->
<!-- {: class="table-of-content"}
* TOC
{:toc} -->

# Main Content

<!-- ## Turorials given by Sense Time on Bilibili
Sense Time is a major collaborator with CUHK to develop the MMDetection. They posted a series of tutorials on Bilibili.
The tutorials introduce two-staged models -- RCNN, Fast RCNN, Faster RCNN, FPN, and how to use the MMDetection Tutorials.
Neck: FPN
Head: RPN Head and ROI Head

## Coursera courses on Object Detections by Andrew Ng
ResNet 50: Stacked by identity blocks and convolutional blocks -->

### Object Localization:
Each input image X with a label y = [y1, y2, y3, y4, y5, y6, y7, y8]

where y1 = Pc = 1 if there is an object else 0

y2 = nx, x coordinate of the center of the bounding box

y3 = ny, y coordinate of the center of the bounding box

y4 = nw, width of bounding box

y5 = nh, height of bounding box

y6 = 1 if the object is class 1 else 0

y7 = 1 if the object is class 2 else 0

y8 = 1 if the object is class 3 else 0

### Landmark detection

instead of nx, ny, nw, nh, we use a series of points

p1x, p1y, p2x, p2y,....p64x, p64y to represent a series of dots on a shape, say eye or jawline

### Object Detection

input x = closely cropped image of a car at the center of the image

output y = 1 if this is a car else 0

sliding windows detection

larger sliding window

even larger sliding window

disadvantage: huge computational cost

solution: convolutional sliding window detection

Convolutional Implementation of Sliding Window Detection

Turn FC layer into convolutional layer

### Bounding Box Predictions

YOLO algorithms

Intersection over Union (IoU)

Non-max suppression: detect the object only once, pick the bounding box with the largest IoU

Anchor-boxes

### Region Proposal

R-CNN: Regions CNN

segmentation algorithm

R-CNN: Propose regions, classify proposed regions one at a time. Output label + bounding box.

Downside: quite slow

- classify one region at a time since not convolutional implementation

Fast R-CNN: Propose Regions. Use convolutional implementation of sliding windows to classify all proposed regions.

- classify all regions all at once with convolutional implementation of sliding windows.
- but the Propose Region part is still slow

Faster R-CNN: Use convolutional network to propose regions.

Andrew Ng claims Faster R-CNN still slower than YOLO algorithm.

### Semantic Segmentation
per-pixel class labels

U-Net

Transpose Convolutions

U-Net Architecture

make a 2 by 2 into 4 by 4

## Environment Setup
1. In your google drive, create a new .ipynb file for environment setup, here I name it “SETUP.ipynb”. In SETUP.ipynb do the following environment setup
2. First mount your google drive
    
    ```python
    import os
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    
3. Here I create a new directory named “MMDet1” and cd into it
    
    ```python
    ! mkdir MMDet1
    %cd /content/drive/My Drive/MMDet1
    ```
    
4. Now we can use pip to install the mmdetection package and other dependencies
    
    I made some adjustments to the official colab tutorial at:
    
     [https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb)
    
    ```python
    # install dependencies: (use cu101 because colab has CUDA 10.1)
    !pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    
    # install mmcv-full thus we could use CUDA operators
    !pip install mmcv-full
    
    # Install mmdetection
    !rm -rf mmdetection
    !git clone https://github.com/open-mmlab/mmdetection.git
    %cd mmdetection
    
    !pip install -e .
    
    # install Pillow 7.0.0 back in order to avoid bug in colab
    !pip install Pillow==7.0.
    ```
    
    The above installation command does not specify version of MMCV, this could lead to confusing error messages later on. it would also result in an unusually long download time. So here I change the original implementation to specify version details:
    
    ```python
    # install dependencies: (use cu101 because colab has CUDA 10.1)
    !pip install -U torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
    # install mmcv-full thus we could use CUDA operators
    # !pip install mmcv-full 
    
    !pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    
    # Install mmdetection
    !rm -rf mmdetection
    !git clone https://github.com/open-mmlab/mmdetection.git
    %cd mmdetection
    
    !pip install -e .
    
    # install Pillow 7.0.0 back in order to avoid bug in colab
    !pip install Pillow==7.0.0
    ```
    
    Executing the above block takes about 5 minutes for me.
    
5. Next we import the installed packages. (remember to set your runtime to GPU) Print package version to verify that we installed correctly:
    
    ```python
    # Check Pytorch installation
    import torch, torchvision
    print(torch.__version__, torch.cuda.is_available())
    
    # Check MMDetection installation
    import mmdet
    print(mmdet.__version__)
    
    # Check mmcv installation
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(get_compiling_cuda_version())
    print(get_compiler_version())
    ```
    
    if everything is correct, this block should output:
    
    ```python
    1.8.1+cu111 True
    2.21.0
    11.1
    GCC 7.3
    ```
    

## Dataset Preparation
I wrote a colab tutorial on this part at:

[Google Colaboratory](https://colab.research.google.com/drive/1-wVLMduYdDXlFtgUIhhWxpTCWwOexXa8#scrollTo=bKZ_LzeLGkfe)

1. First setup drive environment by installing mmdetection and going into the folder we created last time:
    
    ```python
    %load_ext autoreload
    %autoreload 2
    ```
    
    ```python
    import os
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    ```
    
    ```python
    # install dependencies: (use cu101 because colab has CUDA 10.1)
    #!pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    !pip install -U torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
    # install mmcv-full thus we could use CUDA operators
    # !pip install mmcv-full 
    
    # !!!!!!!!!!remember to install with specific version number!!!!!!!!!!!
    !pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    
    # Install mmdetection
    !rm -rf mmdetection
    !git clone https://github.com/open-mmlab/mmdetection.git
    %cd mmdetection
    
    !pip install -e .
    
    # install Pillow 7.0.0 back in order to avoid bug in colab
    !pip install Pillow==7.0.0
    ```
    
2. Dataset download: COCO-2017

    
    ```python
    # here we download the dataset inside our mmdetection folder
    
    !python3 tools/misc/download_dataset.py --dataset-name coco2017 
    
    # remember that the command only downloads the zip files containing the coco dataset
    # to actually use the dataset, we need to unzip first
    ```
    
3. Unzip the coco dataset. This may take approximately 2 hours, do not close browser during any part of this process!!!!

    
    ```python
    !unzip "data/coco/annotations_trainval2017.zip" -d "data/coco/"
    
    !unzip "data/coco/test2017.zip" -d "data/coco/"
    
    !unzip "data/coco/train2017.zip" -d "data/coco/"
    
    !unzip "data/coco/val2017.zip" -d "data/coco/"
    ```
    
4. Dataset download: Pascal_VOC 2007

    
    ```python
    # here we download the dataset inside our mmdetection folder
    
    !python3 tools/misc/download_dataset.py --dataset-name voc2007
    
    # remember that the command only downloads the zip files containing the coco dataset
    # to actually use the dataset, we need to unzip first
    ```
    
5. Unzip the PASCAL-VOC dataset. This is quicker than the coco dataset due to smaller size and tar over zip, again, remebmder not to close browser during any part of this process!!!!

    
    ```python
    # voc is faster to load because of its smaller size and the fact that it uses tar instead of zip
    
    !tar -xvf "data/voc/VOCdevkit_08-Jun-2007.tar" -C "data/voc/"
    
    !tar -xvf "data/voc/VOCtest_06-Nov-2007.tar" -C "data/voc/"
    
    !tar -xvf "data/voc/VOCtrainval_06-Nov-2007.tar" -C "data/voc/"
    ```
    
6. Now just wait a few hours for google drive to load all the images and you are good to go! You can also run the detection now using the images stored in Cache(wierd drive memory).

## Model Config
### PVT:

- [COCO](https://www.notion.so/COCO-57eedad8519744a09e3f4e207548df5d)
Configurations and Model pairs:

website: [https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt)

PVT: 

coco config: configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py

coco model: checkpoints/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth

config link: [https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b0_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b0_fpn_1x_coco.py)

model link: [https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth)





- [VOC](https://www.notion.so/VOC-c540be0c7862426ca5a4d88840d911f1)
voc retina net config link: [https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py)

voc retina net model  link: [https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth)

voc configs/models

[https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc)

[https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth)

[https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py)

### Faster R-CNN:

- coco:

config: configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
model: checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \

- voc:

config: configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
model: checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth \

### Swin:

config: [https://github.com/open-mmlab/mmdetection/blob/master/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py)

model: [https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth)

### GCNet:

config: [https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco.py)

model: [https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth](https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth)

### VFNet:

config: [https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py)

model: [https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth)

### RPN PyTorch:

config: [https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_1x_coco.py)

model: [https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth)

### YOLOv3:

config: [https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py)

model: [https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth)

first use backbone to extract features

then crop features instaed of pictures

R-FCN: Region-based Fully Convolutional Networks

# **Model**

In MMDetection, model components are basically categorized as 4 types.

- backbone: usually a FCN network to extract feature maps, e.g., ResNet.
- neck: the part between backbones and heads, e.g., FPN, ASPP.
- head: the part for specific tasks, e.g., bbox prediction and mask prediction.
- roi extractor: the part for extracting features from feature maps, e.g., RoI Align.

We also write implement some general detection pipelines with the above components, such as `SingleStageDetector` and `TwoStageDetector`.

(https://mmdetection.readthedocs.io/en/v1.2.0/TECHNICAL_DETAILS.html)


# **Inference with Pre-trained Models**

We provide testing scripts to evaluate a whole dataset (COCO, PASCAL VOC, Cityscapes, etc.), and also some high-level apis for easier integration to other projects.

### **Test a dataset**

- [x]  single GPU testing
- [x]  multiple GPU testing
- [x]  visualize detection results

You can use the following commands to test a dataset.

```python
*# single-gpu testing*
python tools/test.py **${**CONFIG_FILE**}** **${**CHECKPOINT_FILE**}** [--out **${**RESULT_FILE**}**] [--eval **${**EVAL_METRICS**}**] [--show]

*# multi-gpu testing*
./tools/dist_test.sh **${**CONFIG_FILE**}** **${**CHECKPOINT_FILE**}** **${**GPU_NUM**}** [--out **${**RESULT_FILE**}**] [--eval **${**EVAL_METRICS**}**]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `proposal_fast`, `proposal`, `bbox`, `segm` are available for COCO, `mAP`, `recall` for PASCAL VOC. Cityscapes could be evaluated by `cityscapes` as well as all COCO metrics.
- `-show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.

If you would like to evaluate the dataset, do not specify `--show` at the same time.

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `proposal_fast`, `proposal`, `bbox`, `segm` are available for COCO, `mAP`, `recall` for PASCAL VOC. Cityscapes could be evaluated by `cityscapes` as well as all COCO metrics.
- `-show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment. Otherwise, you may encounter an error like `cannot connect to X server`.
- `-show-dir`: If specified, detection results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
- `-show-score-thr`: If specified, detections with scores below this threshold will be removed.
- `-cfg-options`: if specified, the key-value pair optional cfg will be merged into config file
- `-eval-options`: if specified, the key-value pair optional eval cfg will be kwargs for dataset.evaluate() function, it's only for evaluation

https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md

# **COCO/mAP**
https://www.notion.so/COCO-mAP-ca19abad1683422b89773ee4ded383d6#7eb5774dd0e94084be0f9ff5a90574ce

# **Config Name Style**

**We follow the below style to name config files. Contributors are advised to follow the same style.**

```python
**{model}_[model** **setting]_{backbone}_{neck}_[norm** **setting]_[misc]_[gpu** **x** **batch_per_gpu]_{schedule}_{dataset}**
```

**`{xxx}` is required field and `[yyy]` is optional.**

- **`{model}`: model type like `faster_rcnn`, `mask_rcnn`, etc.**
- **`[model setting]`: specific setting for some model, like `without_semantic` for `htc`, `moment` for `reppoints`, etc.**
- **`{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).**
- **`{neck}`: neck type like `fpn`, `pafpn`, `nasfpn`, `c4`.**
- **`[norm_setting]`: `bn` (Batch Normalization) is used unless specified, other norm layer type could be `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization). `gn-head`/`gn-neck` indicates GN is applied in head/neck only, while `gn-all` means GN is applied in the entire model, e.g. backbone, neck, head.**
- **`[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `albu`, `mstrain`.**
- **`[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.**
- **`{schedule}`: training schedule, options are `1x`, `2x`, `20e`, etc. `1x` and `2x` means 12 epochs and 24 epochs respectively. `20e` is adopted in cascade models, which denotes 20 epochs. For `1x`/`2x`, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs. For `20e`, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.**
- **`{dataset}`: dataset like `coco`, `cityscapes`, `voc_0712`, `wider_face`.**

# MMdetection Structure

## **We basically categorize model components into 5 types.**

1. **backbone:** usually an FCN network to extract feature maps, e.g., ResNet, MobileNet.
2. **neck:** the component between backbones and heads, e.g., FPN, PAFPN.
3. **head:** the component for specific tasks, e.g., bbox prediction and mask prediction.
4. **ROI extractor:** the part for extracting RoI features from feature maps, e.g., RoI Align.
5. **loss:** the component in head for calculating losses, e.g., FocalLoss, L1Loss, and GHMLoss.


# **Backbone, Neck, Head**

There is a popular paradigm for deep learning-based object detectors: the backbone network (typically designed for classification and pre-trained on ImageNet) extracts basic features from the input image, and then the neck (e.g., feature pyramid network) enhances the multi-scale features from the backbone, after which the detection head predicts the object bounding boxes with position and classification information. Based on detection heads, the cutting edge methods for generic object detection can be briefly categorized into two major branches. The first branch contains one-stage detectors such as YOLO, SSD, RetinaNet, NAS-FPN, and EfficientDet. The other branch contains two-stage methods such as Faster R-CNN, FPN, Mask RCNN, Cascade R-CNN, and Libra R-CNN. Recently, academic attention has been geared toward anchor-free detectors due partly to the emergence of FPN and focal Loss, where more elegant end-to-end detectors have been proposed. On the one hand, FSAF, FCOS, ATSS and GFL improve RetinaNet with center-based anchor-free methods. On the other hand, CornerNet and CenterNet detect object bounding boxes with a keypoint-based method.

https://arxiv.org/pdf/2107.00420.pdf



## **Models:**

We first categorize all the models into two categories based on their detection heads, there are: 1-stage detection model and 2-stage detection models. Then, we evaluate performances within each category before comparing the overall performances of the two model categories.

At last we can investigate the development of models in chronological order.

Popular models that we plan to investigate are listed below:

- 1-stage detection models
    
    YOLO
    
    YOLOv2

    YOLOv3
    
    SSD
    
    DSSD
    
    RetinaNet
    
    NAS-FPN
    
    EfficientDet

    
- 2-stage detection models
    
    R-CNN
    
    Fast-R-CNN
    
    Faster-R-CNN
    
    FPN
    
    Mask-R-CNN
    
    Cascade-R-CNN
    
    Libra R-CNN
    
    SPP-Net
    

## **Backbones:**

The backbone of the model used for feature extraction has a large influence over model performance

- CNN-Based Backbones:
    - VGG (ICLR'2015)
    - ResNet (CVPR'2016)
    - ResNeXt (CVPR'2017)
    - MobileNetV2 (CVPR'2018)
    - [HRNet (CVPR'2019)](https://github.com/open-mmlab/mmdetection/blob/master/configs/hrnet)
    - [Generalized Attention (ICCV'2019)](https://github.com/open-mmlab/mmdetection/blob/master/configs/empirical_attention)
    - [GCNet (ICCVW'2019)](https://github.com/open-mmlab/mmdetection/blob/master/configs/gcnet)
    - [Res2Net (TPAMI'2020)](https://github.com/open-mmlab/mmdetection/blob/master/configs/res2net)
    - [RegNet (CVPR'2020)](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet)
    - [ResNeSt (ArXiv'2020)](https://github.com/open-mmlab/mmdetection/blob/master/configs/resnest)
- Transformer-Based Backbones:
    - [Swin (CVPR'2021)](https://github.com/open-mmlab/mmdetection/blob/master/configs/swin)
    - [PVT (ICCV'2021)](https://github.com/open-mmlab/mmdetection/blob/master/configs/pvt)
    - [PVTv2 (ArXiv'2021)](https://github.com/open-mmlab/mmdetection/blob/master/configs/pvt)

## **Necks:**

enhances the multi-scale features from the backbone

- [PAFPN (CVPR'2018)](https://github.com/open-mmlab/mmdetection/blob/master/configs/pafpn)
- [NAS-FPN (CVPR'2019)](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn)
- [CARAFE (ICCV'2019)](https://github.com/open-mmlab/mmdetection/blob/master/configs/carafe)
- [FPG (ArXiv'2020)](https://github.com/open-mmlab/mmdetection/blob/master/configs/fpg)
- [GRoIE (ICPR'2020)](https://github.com/open-mmlab/mmdetection/blob/master/configs/groie)

## **Heads:**

predicts the object bounding boxes with position and classification information

- one stage detection head
- two stage detection head

I believe detection head corresponds to model selection



# Adversarial Attack

# Black-patches attack

Fast RCNN

```python
!mkdir checkpoints
!wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
```

```python
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cuda:0')
```

```python
import matplotlib.pyplot as plt
import mmcv
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# Use the detector to do inference
img = mmcv.imread('demo/demo.jpg')
plt.figure(figsize=(15, 10))
plt.imshow(mmcv.bgr2rgb(img))
# print(img.shape) #(427, 640, 3)

# Draw a Rectangle in
image = np.zeros((512,512,3), np.uint8)
# cv2.rectangle(image, (100,100), (300,250), (127,50,127), -1)
cv2.rectangle(img, (310,220), (360,270), (0,0,0), -1)
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()

result = inference_detector(model, img)
```

```python
# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)
```

## Fast RCNN specific positions of black patches and result:

### experiment 1:

```python
cv2.rectangle(img, (310,220), (360,270), (0,0,0), -1)
```

![Untitled]()

### experiment 2:

```python
cv2.rectangle(img, (100,100), (300,250), (127,50,127), -1)
```

![Untitled]({{ '/assets/images/team09/1.jpg' | relative_url }}){: style="width: 400px; max-width: 100%;"}*Fig 1. YOLO: An object detection method in computer vision* [1]

### experiment 3:

```python
cv2.rectangle(img, (300,220), (380,270), (0,0,0), -1)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2f2cc80c-fd1e-44a9-96d6-c9d4a3c8bf6a/Untitled.png)

### experiment 4:

```python
cv2.rectangle(img, (280,200), (400,300), (0,0,0), -1)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2aff22dd-9045-448f-b570-c0ce68f876b2/Untitled.png)

### experiment 5:

```python
cv2.rectangle(img, (260,190), (410,320), (0,0,0), -1)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b51a246-7d6d-44ad-a627-c42769055d80/Untitled.png)

### experiment 6:

experiment using our living room’s picture

![315_fast_rcnn_result.jpeg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b4b3836-6c87-4d18-9f9e-bee70eb32293/315_fast_rcnn_result.jpeg)

### experiment 6:

```python
cv2.rectangle(img, (700,800), (1000,1000), (0,0,0), -1)
```

network is fooled to recognize our black patch as a “TV” with score = 0.63




# Evaluation Code

## GCNet.ipynb
https://colab.research.google.com/drive/1dIadqBjKxbT2E-yby1YraMAMLmzhBHBg
## Faster R-CNN.ipynb
https://colab.research.google.com/drive/14YV0cGt2tOMpZeNS6V-mN1eNGaXWoNSf
## PVT-v2.ipynb
https://colab.research.google.com/drive/1Novfkkgq5qQQM3-NRmqEpyWEcxc3b7_p
## RPN.ipynb
https://colab.research.google.com/drive/1TyKwO_QXXQqFhu135sPNM0YreksATOLO?usp=sharing
## Swin.ipynb
https://colab.research.google.com/drive/1aggq_JyvM4JpVp_EvsPkikYBwki_gxUe
## VFNet.ipynb
https://colab.research.google.com/drive/1dj94bQ_8u3qzoChjg3KrOcJMshtzNmc_?usp=sharing
```python
!python3 tools/test.py \
    configs/vfnet/vfnet_r50_fpn_1x_coco.py \
    checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth  \
    --eval bbox proposal \
    --eval-options "classwise=True"
```
 <p> /content/mmdetection/mmdet/utils/setup_env.py:33: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting OMP_NUM_THREADS environment variable for each process '
/content/mmdetection/mmdet/utils/setup_env.py:43: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting MKL_NUM_THREADS environment variable for each process '
loading annotations into memory...
Done (t=2.28s)
creating index...
index created!
load checkpoint from local path: checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth
[>>] 5000/5000, 4.0 task/s, elapsed: 1239s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=1.29s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=46.34s).
Accumulating evaluation results...
DONE (t=13.19s).

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.587
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.757


+---------------+-------+--------------+-------+----------------+-------+ 
| category      | AP    | category     | AP    | category       | AP    | 
|---------------|-------|--------------|-------|----------------|-------| 
| person        | 0.557 | bicycle      | 0.305 | car            | 0.445 | 
| motorcycle    | 0.407 | airplane     | 0.654 | bus            | 0.644 | 
| train         | 0.599 | truck        | 0.375 | boat           | 0.274 | 
| traffic light | 0.275 | fire hydrant | 0.682 | stop sign      | 0.632 | 
| parking meter | 0.461 | bench        | 0.219 | bird           | 0.352 | 
| cat           | 0.654 | dog          | 0.620 | horse          | 0.583 | 
| sheep         | 0.498 | cow          | 0.585 | elephant       | 0.637 | 
| bear          | 0.704 | zebra        | 0.679 | giraffe        | 0.676 | 
| backpack      | 0.169 | umbrella     | 0.403 | handbag        | 0.123 | 
| tie           | 0.325 | suitcase     | 0.389 | frisbee        | 0.655 | 
| skis          | 0.239 | snowboard    | 0.274 | sports ball    | 0.462 | 
| kite          | 0.428 | baseball bat | 0.273 | baseball glove | 0.349 | 
| skateboard    | 0.509 | surfboard    | 0.335 | tennis racket  | 0.485 | 
| bottle        | 0.394 | wine glass   | 0.386 | cup            | 0.434 | 
| fork          | 0.310 | knife        | 0.162 | spoon          | 0.136 | 
| bowl          | 0.413 | banana       | 0.239 | apple          | 0.209 | 
| sandwich      | 0.358 | orange       | 0.335 | broccoli       | 0.237 | 
| carrot        | 0.225 | hot dog      | 0.307 | pizza          | 0.495 | 
| donut         | 0.461 | cake         | 0.354 | chair          | 0.270 | 
| couch         | 0.405 | potted plant | 0.288 | bed            | 0.326 | 
| dining table  | 0.178 | toilet       | 0.585 | tv             | 0.558 | 
| laptop        | 0.585 | mouse        | 0.613 | remote         | 0.284 | 
| keyboard      | 0.534 | cell phone   | 0.344 | microwave      | 0.576 | 
| oven          | 0.316 | toaster      | 0.372 | sink           | 0.360 | 
| refrigerator  | 0.533 | book         | 0.148 | clock          | 0.511 | 
| vase          | 0.386 | scissors     | 0.261 | teddy bear     | 0.469 | 
| hair drier    | 0.093 | toothbrush   | 0.234 | None           | None  | 
+---------------+-------+--------------+-------+----------------+-------+

Evaluating proposal...
Loading and preparing results...
DONE (t=0.56s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=75.46s).
Accumulating evaluation results...
DONE (t=12.70s).

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.678
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.486
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.296
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.522
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.691
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.820

OrderedDict([('bbox_mAP', 0.408), ('bbox_mAP_50', 0.587), ('bbox_mAP_75', 0.441), ('bbox_mAP_s', 0.244), ('bbox_mAP_m', 0.45), ('bbox_mAP_l', 0.526), ('bbox_mAP_copypaste', '0.408 0.587 0.441 0.244 0.450 0.526'), ('mAP', 0.453), ('mAP_50', 0.678), ('mAP_75', 0.486), ('mAP_s', 0.296), ('mAP_m', 0.522), ('mAP_l', 0.633)])
</p>
## YOLOv3
https://colab.research.google.com/drive/178sU_2FkqzRww05knIigsLRrA_ed7P82?usp=sharing
<!-- Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md) -->

<!-- ## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work. -->

<!-- 
### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          | -->



<!-- ### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
``` -->


<!-- ### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/). -->

## Reference

[1] Liu, Ze, et al. "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[2] Wang, Wenhai, et al. "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction Without Convolutions." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[3] He et al. "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016.
[4] Wang et al. "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[5] Wang et al. "Pvtv2: Improved baselines with pyramid vision transformer." *arXiv preprint arXiv:2106.13797*. 2021.
[6] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." *arXiv preprint arXiv:2004.08955*. 2021.
[7] Gao, Shang-Hua, et al. "Res2Net: A New Multi-Scale Backbone Architecture." *Proceedings of the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*. 2021.
[8] Chen, Kai, et al. "MMDetection: Open MMLab Detection Toolbox and Benchmark. " *arXiv preprint arXiv:1906.07155*. 2019.
---



## Code Repository

[MMDetection Main](https://github.com/open-mmlab/mmdetection)

[PVT and PVTv2](https://github.com/whai362/PVT)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[ResNeSt](https://github.com/zhanghang1989/ResNeSt)

[Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels)

[ResNet](https://github.com/KaimingHe/deep-residual-networks)