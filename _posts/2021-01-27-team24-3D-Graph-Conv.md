---
layout: post
comments: true
title: Enhanced Self-Driving with combination of map and lidar inputs.
author: Alexander Swerdlow, Puneet Nayyar
date: 2022-01-27
---


> 3D Point Cloud understanding is critical for many robotics applications with unstructured environments such as self-driving cars. LIDAR sensors provide accurate, high-resolution point clouds that provide a clear view of the environment but making sense of this data can be computationally expensive and is generally difficult. Graph convolutional networks aim to exploit geometric information in the scene that is difficult for 2D based image recognition approaches to reason about.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Our project seeks to implement and improve upon graph convolution approaches to 3D point cloud classification and segmentation. We plan to use the dynamic graph approach proposed by [1] which creates a graph between each layer with connections between nearby points and uses an asymmetric edge kernel that incorporates relative and absolute vertex locations.

## Implementation

## Demo

## Reference

[1] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph CNN for learning on point clouds,” ACM Transactions on Graphics, vol. 38, no. 5, pp. 1–12, 2019.

[2] X. Wei, R. Yu, and J. Sun, “View-GCN: View-based graph convolutional network for 3D shape analysis,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[3] Q. Xu, X. Sun, C.-Y. Wu, P. Wang, and U. Neumann, “Grid-GCN for fast and Scalable Point Cloud Learning,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[4] Y. Chai, P. Sun, J. Ngiam, W. Wang, B. Caine, V. Vasudevan, X. Zhang, and D. Anguelov, “To the point: Efficient 3D object detection in the range image with graph convolution kernels,” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[5] H. Haotian,  F. Wang and H. Le. “VA-GCN: A Vector Attention Graph Convolution Network for learning on Point Clouds.” ArXiv abs/2106.00227 2021.

[6] L. Chen and Q. Zhang, “DDGCN: Graph convolution network based on direction and distance for point cloud learning,” The Visual Computer, 2022.

## Code Repository
[1] [Dynamic Graph CNN for Learning on Point Clouds](https://github.com/WangYueFt/dgcnn)

[2] [Pytorch code for view-GCN](https://github.com/weixmath/view-GCN)

[3] [Grid-GCN for Fast and Scalable Point Cloud Learning](https://github.com/Xharlie/Grid-GCN)

[4] [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)

---
