---
layout: post
comments: true
title: Enhanced Self-Driving with combination of map and lidar inputs.
author: Zifan Zhou, Justin Cui
date: 2022-01-18
---


> Self-driving is a hot topic for deep vision applications. However Vision-based urban driving is hard. Lots of methods for learning to drive have been proposed in the past several years. In this work, we focus on reproducing "Learning to drive from a world on rails" and trying to solve the drawbacks of the methods such as high pedestrian friction rate. We will also utilize the lidar(which is preinstalled on most cars with some self-driving capability) data to achieve better bench mark results.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Our work is a continuation of [Learning to drive from a world on rails](https://dotchen.github.io/world_on_rails/) which uses dynamic programming to learn an agent from past driving logs and then applies it to generate action values. This work is also very closely related to [Learning by Cheating](https://github.com/dotchen/LearningByCheating) which uses a similar two-stage approach to tackle the learning problem. At the same time, we will compare our work with a few others that uses a totally different approach such as [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://github.com/aidriver/ChauffeurNet) that learns to drive by synthesizing images to deal with the worst case scenario. In our work, we will try to solve the drawbacks of "Learning to drive from a world on rails" which doesn't use the most accurate Lidar data and has a high pedestrian friction rate problem.

## Implementation

## Demo

## Reference
Please make sure to cite properly in your work, for example:

[1] Chen, Koltun, et al. "Learning to drive from a world on rails" *Proceedings of the International Conference on Computer Vision*. 2021.

[2] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Learning by cheating. In CoRL, 2019.

[3] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. In RSS, 2019.

## Code Repository
[1] [Learning to drive from a world on rails](https://dotchen.github.io/world_on_rails/)

[2] [Learning by Cheating](https://github.com/dotchen/LearningByCheating)

[3] [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://github.com/aidriver/ChauffeurNet)

---
