---
layout: post
comments: true
title: Video Colorization
author: Tony Xia, Vince Ai
date: 2022-01-28
---


> Historical videos like old movies are all black and white before the invention of colored cameras. However, have you wondered how the good old time looked like with colors? We will attempt to colorize old videos with the power of deep generative models.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Colorization is the process of estimating RGB colors from grayscale images or video frames to improve their aesthetic and perceptual quality. Image colorization has been one of the hot-pursuit problems in computer vision research in the past decades. Various models has be proposed that can colorize image in increasing accuracy. Video colorization, on the other hand, remains relatively unexplored, but has been gaining increasingly popularity recently. It inherits all the challenges faced by image colorization, while adding more complexities to the colorization process. 

In this project, we will cover some of the state-of-the-art models in video colorization, examine their model architecture, explain their methodology of tackling the problems, and compare their effectiveness. We will also identify the limitations of current models, and shed light on the future course of research in this field. 

## Problem Formulation
Formally, video colorization is the problem where given a sequence of grayscale image frame as input, the model aims to recover the corresponding sequence of RGB image frame. To simplify the problem, we usually adopts YUV channels, where only 2 channels needs to be predicted instead of 3. Video colorization poses the following challenges to the research community:

    1. Like image colorization, it is a severely ill-posed problem, as two of the 3 channels are missing. They have to be inferred from other sementics of the image like object detection.

    2. Unlike image colorization that only needs to take care of one image at a time, video colorization needs to remain temporally consistent when colorizatoin a sequence of video frames. Directly applying image colorization methods to videos will cause flickering effects. 

    3. While image colorization is stationary, video colorization has to deal with dynamics scenes, so some frames will be blurred and hard to colorize.

    4. Video colorization requires much more computing power than image colorization as the dataset are usually huge and the models are more complex.
## Models
 
### Learning Blind Video Temporal Consistency  (2018)
### Deep Exemplar-based Video Colorization (2019)
### Framewise Instance Aware Image Colorization (2020)
InstColor proposed novel network architecture that leverages off-the-shelf models to detect the object and learn from large- scale data to extract image features at the instance and full-image level, and to optimize the feature fusion to obtain the smooth colorization results. The key insight is that a clear figure-ground separation can dramatically improve colorization performance.

The model consists of three parts:

    1. an off-the-shelf pretrained model to detect object instances and produce cropped object images
    
    2. two backbone networks trained end-to-end for instance, and full-image colorization 
    
    3. a fusion module to selectively blend features extracted from different layers of the two colorization networks

![InstColor]({{ '/assets/images/team12/instColorModel.png' }})

### Temporally Consistent Video Colorization (TCVC) (2021)

## Conclusion

![YOLO]({{ '/assets/images/team12/example.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Deep Exemplar-based Video Colorization* [3].




## Reference
Please make sure to cite properly in your work, for example:

[1] Liu, Yihao, et al. "Temporally Consistent Video Colorization with Deep Feature Propagation and Self-regularization Learning." arXiv preprint arXiv:2110.04562 (2021).  
[2] Anwar, Saeed, et al. "Image colorization: A survey and dataset." arXiv preprint arXiv:2008.10774 (2020).  
[3] Zhang, Bo, et al. "Deep exemplar-based video colorization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.  

---
