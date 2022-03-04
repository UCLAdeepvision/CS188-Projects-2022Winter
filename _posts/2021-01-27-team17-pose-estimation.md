---
layout: post
comments: true
title: Pose Estimation
author: Guofeng Zhang, Zihao Dong
date: 2022-01-27
---


> Human pose estimation is a research area that is closely related to cutting-edge Metaverse and VR techniques. In this blog, we will record the technical details of reproducing an existed work and make small modification to apply the model to more complicated environment to generate pose estimation. We are also considering collecting datasets from other species and train model apply on animals.  


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Our project is based on previous work [Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation](https://zhengyiluo.github.io/projects/kin_poly/). In this project, our goal is to present a object-aware 3D human pose estimation that is able to reconstruct 3D pose and interaction from egocentric videos. The model contains mainly 2 aspects, the universal humanoid controller, which is capable of mimicing human motion with high fidelity through [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), and dynamics-regulated kinematic policy, which receives feedback from physical simulation and thus assist our model training. After the reconstruction of the existed work, we plan to train and apply the model to more complicate situation with interactions with multiple objects at the same time. Below is an indication of how the model should work


## Main content
### Demo


### Our Code and Execution




### Relevant papers and their repo:
1. Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation<br>
    [GitHub Link](https://github.com/KlabCMU/kin-poly)
2. Integral Human Pose Regression<br>
    [GitHub Link](https://github.com/JimmySuen/integral-human-pose)
3. Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image<br>
    [GitHub Link](https://github.com/RhythmJnh/TF_SMPL)

## Reference


---
