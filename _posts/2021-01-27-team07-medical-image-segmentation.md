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
*Fig 1. Automatic Segmentation of Pulmonary Lobes* [1].

## 1606.04797: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation [[Paper](https://arxiv.org/abs/1606.04797)] [[Code](https://github.com/faustomilletari/VNet)] 

Convolutional neural networks have been popular for solving problems in medical image analysis. Most of the developed image segmentation approaches only operate on 2D images. However, medical imaging often consists of 3D volumes, which gives opportunity to the development of CNNs in 3D image segmentation. In their paper V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, Milletari et al propose the V-Net: a volumetric, fully convolutional neural network trained on 3D MRI scans of the prostate to predict segmentation for the entire volume.

Importance: "Segmentation is a highly relevant task in medical image analysis. Automatic delineation of organs and structures of interest is often necessary to perform tasks such as visual augmentation [10], computer assisted diagnosis [12], interventions [20] and extraction of quantitative indices from images [1]. In particular, since diagnostic and interventional imagery often consists of 3D images, being able to perform volumetric segmentations by taking into account the whole volume content at once, has a particular relevance."

Unlike prior approaches for processing 3D image inputs slice-wise (with 2D convolutions on each slice), the proposed model uses volumetric (3D) convolutions. The following figure is a schematic representation of the V-Net model. The left side of the figure depicts a compression path, while the right side depicts a decompression path to the original signal size. Each intermediate "stage" consists of 2-3 convolutional layers. Moreover, the input of each stage is "(a) used in the convolutional layers and processed through the non-linearities and (b) added to the output of the last convolutional layer of that stage in order to enable learning a residual function". The design of this neural network ensures a runtime much faster than similar networks that do not learn residual functions. 

Along the compression path, the data resolution is reduced with convolution kernels of dimensions 2 × 2 × 2 applied at stride 2, effectively halving the size of feature maps at each stage. Additionally, the number of feature channels and in turn number of feature maps doubles at each compression stage. Conversely, the decompression stages apply de-convolutions to increase the data size again, while also accounting for previously extracted features from the left path. The final predictions, after a softmax layer, comprises two volumes outputting the probability of each voxel to belonging to foreground and to background.

The objective function used by the V-Net architecture considers the dice coefficient, a quantity ranging from 0 to 1. The dice coefficient between two binary volumes is defined as 

$$ D = \dfrac{2 \sum_{i}^{N} p_i g_i}{ \sum_{i}^{N} p_i^{2} + \sum_{i}^{N} g_i^{2} }$$

![V-Net Architecture]({{ '/assets/images/team07/160604797arch.png' | relative_url }})
*Fig 2. V-Net Architecture* [1].


## 2104.07961: Advanced Deep Networks for 3D Mitochondria Instance Segmentation [[Paper](https://arxiv.org/abs/2104.07961)] [[Code](https://github.com/Limingxing00/MitoEM2021-Challenge)] 
### Motivation
This paper is inspired by the MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation. There has been an increasingly pressing need for a new automatic mitochondria segmentation approach as current algebraic segmentation methods offeres a rather limited generalizability. As the volume of electron microscopy images increases to the scale of Terabyte, the focus turns to CNN-based approaches. In general, the CNN-based segmentation methods could be categorized into two groups, 
1. Top-down method that uses Mask-RCNN for instance segmentation. 
2. Bottom-up method that predicts a binary segmentation mask, an affinity map or a binary mask with the instance boundary, after which a post-processing algorithm is applied to distinguish instances. 

The authors of this paper decided to approach the problem with a bottom-up method as the elongated and distorted nature of mitochondria makes it difficult to set a proper anchor size for Mask-RCNN.

### Pre-processing, Fine-Tuning, and other Implementation Details
1. Data augmentation is employed; refer to [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7713709/) for more. 
2. Interpolation network is adopted. The interpolation network takes two adjacent frames of the noisy frame as input and predict two kernels; the two adjacent frames are then convolved with the two adjacent frames, the sum of which contributes to the restored frame.

![Interpolation Network]({{ '/assets/images/team07/2104.07961inter.jpeg' | relative_url }}) ![Notation Key]({{ '/assets/images/team07/2104.07961key.jpeg' | relative_url }})
*Fig 3. Interpolation Network* [2]. 

3. The authors devised a two-stage training which they call multi-scale training. Originally, the network is trained in 20K interations with input size 32x256x256 to select the best model; after the model is selected, the input size is changed to 26x320x320 and fine-tuned in 10K iterations. 
4. Emperically, the authors decided on optimizer Adam with a fixed rate of 1e-4 and batch size of 2.
5. The adopted evalution metric is [3D AP-75 metric](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7713709/); in this case, at least 0.75 intersection over the union with the ground truth is required to be a true positive. The jacaard-index coefficient and dice similarity coefficient are evaluated for the foreground objects in the volumes. 

### Architecture
The paper proposes two deep residual networks for the rat and human sample: Res-UNet-R and Res-UNet-H, in that respective order. Interestingly, the Res-UNet-H only differs from Res-UNet-R because it has an additional decoder path. The additional decoder path allows Res-UNet-H to predict the semantic mask and instance boundary separately, which, along with the noise reduction method that will be introduced later, is used to alleviate the influence of noise on segmentation, which is subjectively significant in the human sample. For this reason, this post will be focusing mostly on Res-UNet-H. 
- Convolution Block: since the MitoEM dataset has anisotropic resolution, the authors designed an Anisotropic Convolution Block (ACB) that consists of a 1x3x3 convolutional layer, an exponential linear unit (ELU), two 3x3x3 conventional layers, each of which are followed by an ELU, and a skip connection in the two 3x3x3 conventional layers, as illustrated below. The conventional layers are the used to enlarge the receptive field.
- Network Structure: the two proposed networks are largely inspired by 3D U-Net. The feature maps extracted from a 3D block with a 1x5x5 conventional layer are embedded. There is an ACB to extract the anisotropic information in each layer of the encoder. A conventional layer of 1x3x3 is used to downsample the feature maps in the lateral dimensions. The decoder uses a trilinear upsampling to restore the resolution of the feature maps and ACB is used to reconstruct the detailed information. Two (or one, if using Res-UNet-R), decoder paths are used to predict the semantic mask and the instance boundary separately (simultaneously, if using Res-UNet-R). A visual representation of the network could be seen below.
![Res-UNet-H]({{ '/assets/images/team07/2104.07961resuneth.jpeg' | relative_url }}) ![Notation Key]({{ '/assets/images/team07/2104.07961key.jpeg' | relative_url }})
*Fig 4. Res-UNet-H* [2].


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

$$W_f = \dfrac{\text{sum}(Y_i)}{DHW}$$ is the foreground voxel ratio, D, H, and W, the depth, height, and width of the block, $$ X_M $$ and $$X_B $$ the predicted response maps of the semantic mask and the instance boundary while $$ Y_M $$ and $$ Y_B $$ the corresponding ground-truths of $$ X_M $$ and $$X_B $$.


### Post-Processing algorithm
Once the semantic mask $$X_M$$ and the instance boundary $$X_B$$ are obtained, the seed map is defined as 

$$
S^{j} = \begin{cases} 
      1 & X^{j}_{M} > T_{1}, X^{j}_{B} > T_{2} \\
      0 & \text{else}
   \end{cases}
$$

where $$T_{1}, T_{2}$$ are empirically defined as 0.9 and 0.8 respectively. Note that $$X_M \in \mathbb{R}^{D \cdot H \cdot W}, \ X_B \in \mathbb{R}^{D \cdot H \cdot W}, \ j \in [1, \mathbb{R}^{D \cdot H \cdot W}]$$

<!-- ### Keywords
- Mask-RCNN
- 3D U-Net
- foreground voxel ratio
- Anisotropic?
- semantic mask?
- instance boundary? -->

## 1902.06362: Automatic Segmentation of Pulmonary Lobes Using a Progressive Dense V-Network [[Paper](https://arxiv.org/abs/1902.06362)]
### Motivation
In the medical imaging analysis field, fast and reliable segmentation of lung lobes is important for diagnosis, assessment, and quantification of pulmonary diseases. Existing techniques for segmentation have been slow and not fully automated. To address the lack of an adequate imaging system, this paper proposes a new approach for lung lobe segmentation: a progessive dense V-network (PDV-Net) that is robust, fast, and fully automated.
Some challenges to develop efficient automated system to identify lung fissures are as follows:
1. Fissures are usually incomplete, not extending to the lobar boundaries. 
2. Visual characteristics of lobar boundaries vary in the presence of pathologies. 
3. Other fissures may be misinterpreted as the major or minor fissures that separate the lung lobes.
The authors discuss prior deep learning developments in the medical image segmentation field, such as the dense V-network and progressive holistically nested networks. These approaches were slow and generally low in performance for pathological cases. The proposed PDV-Net model mitigates these limitations, and uses the following architecture.

### Architecture 
The input is down-sampled and concatenated with a convolution of the input, where the convolution has 24 kernels of size 5 x 5 x 5 and stride 2. This is passed onto three dense feature blocks: one block with 5 and two blocks with 10 densely connected convolutional layers. The three dense blocks have growth rates 4, 6, and 8, respectively. All of their convolutional layers have 3 x 3 x 3 kernels and are followed by batch normalization and PReLU.

The dense block outputs are consecutively forwarded in low and high resolution passes via compression and skip connections, which enables the generation of feature maps at three different resolutions. The outputs of the skip connections of the second and third dense feature blocks are decompressed to match the first skip connection's output size. The merged feature maps from the skip connections are passed to a convolutional layer followed by a softmax, to output the final probability maps. This proposed architecture progressively improves the outputs from previous pathways to output the final segmentation result. Similar to the V-Net solution, it utilizes a dice-based loss function for model training.
To train the network, the training volumes are first normalized and rescaled to 512 x 512 x 64. Spatial batch normalization and dropout are incorporated for regularization. Moreover, the authors use the Adam optimizer with learning rate 0.01 and weight decay 10-7.

![Notation Key]({{ '/assets/images/team07/1902.06362arch.png' | relative_url }})
*Fig 3. Architecture for Progressive Dense V-Network* [2]. 

### Results
Using the dice score for comparison, the authors find that the PDV-Net significantly outperforms the 2D U-Net model and the 3D dense V-Net model. Further investigations show that the PDV-Net is robust against the reconstruction kernel parameters, different CT scan vendors, and presence of lung pathologies. Additionally, the model takes approximately 2 seconds segment lung robes from a single CT scan, which is faster than all prior solutions at the time of its proposal. 

## [Code](https://drive.google.com/drive/folders/1uUNNbixXDdgpa9GIWPccQO8RGyPLAviw)


## References
[1] Milletari, Fausto, Nassir Navab, en Seyed-Ahmad Ahmadi. “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation”. arXiv [cs.CV] 2016. 

[2] Li, Mingxing et al. “Advanced Deep Networks for 3D Mitochondria Instance Segmentation”. arXiv [cs.CV] 2021. 

[3] Imran, Abdullah-Al-Zubaer et al. “Automatic Segmentation of Pulmonary Lobes Using a Progressive Dense V-Network”. Lecture Notes in Computer Science (2018): 282–290. 

---
