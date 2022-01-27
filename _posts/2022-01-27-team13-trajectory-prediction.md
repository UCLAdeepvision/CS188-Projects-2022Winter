---
layout: post
comments: true
title: Trajectory Prediction
author: Sudhanshu Agrawal, Jenson Choi
date: 2022-01-27
---

> "Behavior prediction in dynamic, multi-agent systems is an important problem in the context of self-driving cars". [Gao, Jiyang et al.](https://arxiv.org/abs/2005.04259). In this blog, we will investigate a few different approaches to tackling this multifaceted problem (and eventually figure out a specific area that we will focus on).

<!--more-->

{: class="table-of-content"}

- TOC
  {:toc}

## Introduction

Self-driving is one of the biggest applications of Computer Vision in industry. Naturally, being able to predict the trajectory of an autonomous vehicle is paramount to the success of self-driving. Our project will be an extension of [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259), which is a hierachical graph neural network architecture that first exploits the spatial locality of individual road components represented by vectors and then models the high-order interactions among all components. Other recent approaches to trajectory prediction primarily utilize convolutional neural networks (CNNs), which is the most widely used model in Computer Vision in recent years.

## Implementation

## Demo

## Reference

Please make sure to cite properly in your work, for example:

[1] Gao, Jiyang, et al. "Vectornet: Encoding hd maps and agent dynamics from vectorized representation." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2020.

## Code Repository

[1] [Reimplement VectorNet](https://github.com/xk-huang/yet-another-vectornet)

---
