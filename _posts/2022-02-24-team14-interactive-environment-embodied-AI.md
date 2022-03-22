---
layout: post
comments: true
title: Interactive Environment, Embodied AI
author: David Watson
date: 2022-02-24
---

> My project is to investigate current trends in Embodied AI development and research, to report on methods of creating an environment, and guide reader into setting up an interactive environment on their own computer using iGibson.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Full github Repo: https://github.com/StanfordVL/iGibson
I will give a brief discussion on the following areas of the program
with a more in depth section on the learning model.

Learning Model
  3 modules:
	AdaptiveNorm2d
	CompletionNet
	Perceptual

Types of Scenes
  Empty
  Indoor
  Stadium

Types of objects
  Particles
  Pedestrians
  Articulated Objects
  Visual Markers

Tasks
  Behaviors
  Point Navigation
  Room rearrangement

Reward system
  Collision
  Reaching Goal

## Reference

[1] Bokui Shen, et al. "iGibson 1.0: A Simulation Environment for Interactive Tasks in Large Realistic Scenes." arXiv 10 Aug 2021 https://arxiv.org/pdf/2012.02924v6.pdf

[2] Peter Anderson, et al. "On Evaluation of Embodied Navigation Agents." arXiv 18 Jul 2018 https://arxiv.org/pdf/1807.06757v1.pdf

[3] F. Xia et al., "Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 713-720, April 2020, doi: 10.1109/LRA.2020.2965078.