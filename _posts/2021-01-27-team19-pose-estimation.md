---
layout: post
comments: true
title: Pose Estimation
author: Nicholas Dean and Ellie Krugler
date: 2022-02-24
---

> (Revise) Pose estimation is defined as the task of using an machine learning model to estimate the pose of a person from an image or a video by estimating the spatial locations of key body joints. In our project, we will use pose estimation to render 2D human body shapes for various photos.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Helpful Info (delete later)
You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Introduction

In this article, we will discuss what pose estimation is, older as well as current models used for pose estimation, dive into the technical details of how these models are implemented, go into possible innovations, and conclude with applicaitons of pose estimation.

## Background

Human pose estimation is a computer vision task that includes detecting and tracking keypoints of the body. Examples of such keypoints are “left elbow,” “right ankle,” or the “chin.”

Pose estimation is a process that requires significant computatinal resouces, and just over a decade ago, such a task seemed near impossible. With recent advances in GPU, TPU, and other processing techology, applications incorporating pose estimation like motion tracking are becoming possible.

First, let us introduce several key terminologies related to pose estimation that will help us understand the remainder of the article.

### Instance Segmentation

Instance segmentation is the process of classifying individual objects and then localizing each of them with a bounding box. This is different from semantic segmentation, which classifies an image pixel by pixel and treats all pixels in the same class as one object.

With a single-object image, the only tasks necessary are classifying, which is giving a class to the main object, and localizing, which is finding the object's bounding box. With multiple-object images, object detection is required to carry this task out for each object in the image and differentiate between different objects.

### Top-down vs. Bottom-up methods

All pose estimation models can be grouped into bottom-up and top-down methods.

* **Top-down** methods detect a person first and then estimate body joints within the detected bounding boxes.
* **Bottom-up** methods estimate each body joint first and then group them to form a unique pose.

### Types of Human Body Modeling

* **Kinematic** model 
    * Also called skeleton-based model
    * Used for 2D and 3D applications
    * Used to capture the relations between different body parts
* **Planar** model
    * Also called contour-based model
    * Used for 2D applications
    * Used to represent the appearance and shape of a human body
    * Body parts are represented by multiple rectangles approximating the human body contours
* **Volumetric** model
    * Used for 3D applications
    * Used to represent the relations between different body parts and the shape of a human body (combination of Kinematic and Planar models)

![The three types of human body models]({{ '/assets/images/team19/types_of_body_models.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. The three types of human body models.*

### Challenges

Many factors make human pose estimation difficult, including abnormal viewing angles, different background contexts, overlapping body parts, and different clothing shapes. The pose estimation method must also account for real-world variations such as lighting and weather. In addition, small joints can be difficult to detect.

### Datasets

The MS COCO dataset is the most popular dataset for human pose estimation. It features 17 different classes, known as keypoints. The classes are listed below. Each keypoint is annotated with three numbers (x,y,v), where x and y mark the coordinates and v indicates if the keypoint is visible. 

```
"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
"left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
"right_knee", "left_ankle", "right_ankle"
```

## Existing Models

Now that we have some intuition behind the background and challenges of pose estimation, let's dive into some of the models that address this task.

### OpenPose

OpenPose is one of the earliest models for human pose estimation and the first real-time multi-person detection system. It will be the main focus of our project.

For its task, OpenPose [1] utilizes Part Affinity Fields, Confidence Maps, and a set of 2D vector fields that encodes the location and orientation of limbs of different people in the image. Confidence Maps are a 2D map representing the confidence that a particular body part is located at any given pixel. OpenPose was the winner of the COCO KeyPoint Detection Challenge in 2016.

OpenPose is a two-part system using Part Affinity Fields and Confidence Maps.

(a): A color image of size w×h as an input image.
(b): A feedforward network simultaneously predicts a set of 2D confidence maps (CM) S of body part locations, and
(c): a set of 2D vector fields L of part affinities, or part affinity fields (PAF), which encode the degree of association between parts
The set S = (S1, S2, …, SJ ) has J confidence maps, one per part.
The set L = (L1, L2, …, LC) has C vector fields, one per limb. Each image location in LC encodes a 2D vector.
(d): Then, the confidence maps and the affinity fields are parsed by greedy inference, and
(e): output the 2D keypoints for all people in the image.

**OpenPose Equations**

Confidence Map of J body parts

$$
S = (S_1, S_2, …, S_J)\ where\ S_j ∈ R^{w×h}, j ∈ \{1 … J\}
$$

Part Affinity Field of C connections between body parts

$$
L = (L_1, L_2, …, L_C)\ where\ L_c ∈ R^{w×hx2}, c ∈ \{1 … C\}
$$

The base network, F, uses feature maps to redefine Lt

$$
L^t = \phi^t(F, L^{t-1}), \forall2 \leq t \leq T_p
$$

New predictions of the confidence maps

$$
S^{T_p} = \rho^t(F, L^{T_p}), \forall t = T_p
$$

$$
S^t = \rho^t(F, L^{T_p}, S^{t-1}), \forall T_p < t \leq T_p + T_c
$$

**Loss Functions**

Part Affinity Fields loss

$$
f^{t_i}_L = \sum^{C}_{c=1} \sum_p W(p) \cdot \left\lVert L^{t_i}_c(p) - L^*_c(p) \right\rVert^2_2
$$

Confidence Map loss

$$
f^{t_k}_S = \sum^{J}_{j=1} \sum_p W(p) \cdot \left\lVert S^{t_k}_j(p) - S^*_j(p) \right\rVert^2_2
$$

**Overall Objective Function**

$$
f = \sum^{T_p}_{t=1} f^t_L + \sum^{T_p + T_c}_{t= T_p + 1} f^t_S
$$

![OpenPose Pipeline]({{ '/assets/images/team19/openpose_pipeline.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. OpenPose Pipeline.*

![OpenPose Network Architecture]({{ '/assets/images/team19/openpose_network_architecture.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 3. OpenPose Network Architecture.*

**OpenPose Confidence Maps**

The confidence maps are a measure of the likelihood of a given body part being in a certain section. For example, the confidence map of the right knee should be 0 in every grid area that a right knee is not present. There is one set, S, for each body part, adding up to J sets that make up the confidence maps.

However, confidence maps are not as crucial to accuracy as Part Affinity Fields are, and some models omit them entirely since they tend to not increase the Average Precision or Average Recall.

**OpenPose Part Affinity Fields**

The Part Affinity Fields connect parts of the body that belong to the same person. For example, if an area is classified as "left_elbow" and another area is classified as "left_wrist," the PAF tells you how likely it is that those two body parts belong to the same person. This proves helpful when images contain multiple people overlapping or standing close to each other, like in crowd situations.

The exact methods of determining this association strength can be done multiple ways, such as:
 * A k-partite graph
 * A bi-partite graph with a greedy algorithm
 * A tree structure
 * Body part classification

### Mask R-CNN

Mask R-CNN is a popular tool for segmenting an image by its pixels (semantically) or by its image objects (instantially). As such, it is not difficult to apply Mask R-CNN to pose estimation.

Once a CNN extracts features from the image, a Region Proposal Network generates bounding box candidates where objects could be. Another layer reduces the features to those of similar size, then runs them in parallel to get segmentation mask proposals. These are used to create binary masks of where an object is and is not present in the image. Finally, keypoints are extracted by segmenting, then combined with the person locations to get a human skeleton for each figure.

This is similar to a top-down approach, except that the person detection step and the body joint estimation step happen in parallel.

### Residual Steps Network (RSN)

Now, we will discuss a modern architecture for pose estimation that was the winner of the COCO KeyPoint Detection Challenge in 2019.

RSN was created with the goal of having the accuracy of the DenseNet model without the huge network capacity requirements. It uses a Residual Steps Block, or RSB, to fuse features with element-wise sum operations rather than concatenation. 

The creators claim that the RSN model outperforms its predecessors on the MS COCO dataset, with better performance in many cases as well.


## Applications

Pose estimation has applications in a number of areas.

* **Human Health and Performance** 
    * Performing motor assessments on patients remotely, especially in pediatrics
        * <em>this can detect disorders that cause atypical development in children</em>
    * Evaluating athletic performances
    * Analyzing gait, especially for people having a stroke or with dementia

* **Driver Safety** 
    * Identifying driver drowsiness or distraction with head pose estimation
    * Predicting cyclists' behavior by their hand signs
    * Vehicle pose estimation

* **Entertainment** 
    * Tracking hand poses instead of having players hold controllers in VR
    * Producing deepfakes of dancing, like [Sway](https://getsway.app/) from Humen.Ai
    * Snapchat's [Lens Studio](https://lensstudio.snapchat.com/), a form of augmented reality


## Reference

[1] Cao, Zhe, et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields" arXiv preprint arXiv:1812.08008 (2019).<br>
[[Paper]](https://arxiv.org/abs/1812.08008) [[Code]](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[2] Luo, Zhengyi, et al. "Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation." arXiv preprint arXiv:2106.05969 (2021).<br>
[[Paper]](https://arxiv.org/abs/2106.05969) [[Code]](https://github.com/KlabCMU/kin-poly)

[3] paper on RSN
[[Paper]](https://arxiv.org/abs/2003.04030) [[Code]](https://github.com/caiyuanhao1998/RSN)

---
