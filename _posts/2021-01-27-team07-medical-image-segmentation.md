---
layout: post
comments: true
title: Medical Image Segmentation
author: Lenny Wu, Katie Chang
date: 2022-01-27
---


> The use of deep learning methods in medical image analysis has contributed to the rise of new fast and reliable techniques for the diagnosis, evaluation, and quantification of human diseases. We will study and discuss various robust deep learning frameworks used in medical image segmentation, such as PDV-Net, Res-UNet-R, and Res-UNet-H. Furthermore, we will implement our own neural network to explore and demonstrate the capabilities of deep learning in this medical field.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Our team explores the applications of CV in healthcare; specifically, notable state-of-the-art medical image segmentation methods. Areas of interest include biomimetic frameworks for human neuromuscular and visuomotor control, segmentation of mitochondria, and segmentation of healthy and pathological lung lobes.


## Example
### Image

![Lubes Segmentation]({{ '/assets/images/team07/seg_lubes.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Automatic Segmentation of Pulmonary Lobes* [1].

## Advanced Deep Networks for 3D Mitochondria Instance Segmentation [[Paper](https://arxiv.org/abs/2104.07961)] [[Code](https://github.com/Limingxing00/MitoEM2021-Challenge)] 
### Motivation
This paper is inspired by the MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation. There has been an increasingly pressing need for a new automatic mitochondria segmentation approach as current algebraic segmentation methods offeres a rather limited generalizability. As the volume of electron microscopy images increases to the scale of Terabyte, the focus turns to CNN-based approaches. In general, the CNN-based segmentation methods could be categorized into two groups, 
1. Top-down method that uses Mask-RCNN for instance segmentation. 
2. Bottom-up method that predicts a binary segmentation mask, an affinity map or a binary mask with the instance boundary, after which a post-processing algorithm is applied to distinguish instances. 

The authors of this paper decided to approach the problem with a bottom-up method as the elongated and distorted nature of mitochondria makes it difficult to set a proper anchor size for Mask-RCNN.

### Pre-processing:
1. Data augmentation is employed
2. interpolation network is adopted. The interpolation network takes two adjacent frames of the noisy frame as input and predict two kernels; the two adjacent frames are then convolved with the two adjacent frames, the sum of which contributes to the restored frame.

### Architecture:
The paper proposes two deep residual networks for the rat and human sample: Res-UNet-R and Res-UNet-H, in that respective order. Interestingly, the Res-UNet-H only differs from Res-UNet-R because it has an additional decoder path. The additional decoder path allows Res-UNet-H to predict the semantic mask and instance boundary separately, which, along with the noise reduction method that will be introduced later, is used to alleviate the influence of noise on segmentation, which is subjectively significant in the human sample. For this reason, this post will be focusing mostly on Res-UNet-H. 
- Convolution Block: since the MitoEM dataset has anisotropic resolution, the authors designed an Anisotropic Convolution Block (ACB) that consists of a 1x3x3 convolutional layer, an exponential linear unit (ELU), two 3x3x3 conventional layers, each of which are followed by an ELU, and a skip connection in the two 3x3x3 conventional layers, as illustrated below. The conventional layers are the used to enlarge the receptive field.
- Network Structure: the two proposed networks are largely inspired by 3D U-Net. The feature maps extracted from a 3D block with a 1x5x5 conventional layer are embedded. There is an ACB to extract the anisotropic information in each layer of the encoder. A conventional layer of 1x3x3 is used to downsample the feature maps in the lateral dimensions. The decoder uses a trilinear upsampling to restore the resolution of the feature maps and ACB is used to reconstruct the detailed information. Two (or one, if using Res-UNet-R), decoder paths are used to predict the semantic mask and the instance boundary separately (simultaneously, if using Res-UNet-R). A visual representation of the network could be seen below.

### Loss
The authors chose to employ a weighted binary cross entropy loss function to combat the class imbalance problem. Both the semantic msak and instance boundary are considered in the loss function as they are both used to generate the post-processing seed map. Thus, the loss function could be described by

$$ L = L_{WBCE}(X_M, Y_M) + L_{WBCE}(X_B, Y_B) $$ 

where 

$$ L_{WBCE}(X_i, Y_i) = \dfrac{1}{DHW} W_{i} \cdot L_{BCE}(X_i, Y_i)$$

$$
W = \begin{cases} 
      Y_i + \dfrac{W_f}{1-W_f}(1-Y_i) & W_f > 0.5 \\
      \dfrac{W_f}{1-W_f}Y_i + 1-Y_i & W_f \leq 0.5
   \end{cases}
$$

$$W_f = \dfrac{\text{sum}(Y_i)}{DHW}$$ is the foreground voxel ratio, D, H, and W, denote the depth, height, and width of the block, and 


### Post-Processing algorithm


<!-- ### Keywords
- Mask-RCNN
- 3D U-Net
- foreground voxel ratio
- Anisotropic?
- semantic mask?
- instance boundary? -->

## Reference
[1] Imran, Abdullah-Al-Zubaer et al. “Automatic Segmentation of Pulmonary Lobes Using a Progressive Dense V-Network”. Lecture Notes in Computer Science (2018): 282–290.

[2] Li, Mingxing et al. “Advanced Deep Networks for 3D Mitochondria Instance Segmentation”. arXiv [cs.CV] 2021. 

[3] M. Nakada, T. Zhou, H. Chen, A. Lakshmipathy and D. Terzopoulos, "Deep Learning of Neuromuscular and Visuomotor Control of a Biomimetic Simulated Humanoid," in IEEE Robotics and Automation Letters, vol. 5, no. 3, pp. 3952-3959, July 2020, doi: 10.1109/LRA.2020.2972829.

---
