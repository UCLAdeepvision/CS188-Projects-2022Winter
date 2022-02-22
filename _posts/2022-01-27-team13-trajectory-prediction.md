---
layout: post
comments: true
title: Trajectory Prediction
author: Sudhanshu Agrawal, Jenson Choi
date: 2022-01-27
---

> Behavior prediction in dynamic, multi-agent systems is an important problem in the context of self-driving cars. In this blog, we will investigate a few different approaches to tackling this multifaceted problem and reproduce the work of [Gao, Jiyang et al.](https://arxiv.org/abs/2005.04259) by implementing VectorNet in PyTorch.

<!--more-->

{: class="table-of-content"}

- TOC
  {:toc}

## Introduction

Self-driving is one of the biggest applications of Computer Vision in industry. Naturally, being able to predict the trajectory of an autonomous vehicle is paramount to the success of self-driving. Our project will be an extension of [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259), which is a hierachical graph neural network architecture that first exploits the spatial locality of individual road components represented by vectors and then models the high-order interactions among all components. Research on trajectory prediction is not limited to the self-driving domain, however, [Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html) and [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892) are more generic examples of work related to multi-agents interaction forecasting which we will also explore in this project. In particular, if possible, we will attempt to extend the work on human trajectory prediction in crowded spaces to simulate the effect that social distancing due to COVID-19 has on the trajectories. We may be able to do this by exploring ways of adding a factor to create a form of 'repulsion' between the agents. 

## Implementation

## Demo

## Reference

Please make sure to cite properly in your work, for example:

[1] Gao, Jiyang, et al. "Vectornet: Encoding hd maps and agent dynamics from vectorized representation." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2020.

[2] Alahi, Alexandre, et al. "Social lstm: Human trajectory prediction in crowded spaces." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

[3] Gupta, Agrim, et al. "Social gan: Socially acceptable trajectories with generative adversarial networks." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2018.

## Code Repository

[1] [Reimplement VectorNet](https://github.com/xk-huang/yet-another-vectornet)

[2] [Social LSTM Implementation in PyTorch](https://github.com/quancore/social-lstm)

[3] [Social GAN](https://github.com/agrimgupta92/sgan)

---
