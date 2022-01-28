---
layout: post
comments: true
title: Terraform, User-Guided 3D Scene Generation
author: Felix Zhang
date: 2022-01-27
---


> This blog will contain updates and technical explanations for Felix Zhang's CS188 Project. The focus of project Terraform is to explore various approaches to controlling and generating 3-D scenes. We are not sure exact approach currently but are actively investigating Generative Neural Radiance Fields as well as Generative Adversial Networks as potential candidates.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Team Paradigm Initial Project Proposal

### Group Members: Felix Zhang

### Topic: Terraform, Guided 3D Scene Generation (with exploration of Generative Neural Radiance Fields and Unsupervised Techniques)

There has been a recent surge in using exploring image generation in three dimensions due to the adoption of virtual, augmented, and mixed reality devices, as well as the possibility of using scenes to power downstream task datasets for agents. We explore in project Terraform various techniques to guide 3D Scene Generation, especially focusing on a recent interesting development of Generation with Neural Radiance Fields.

## Neural Radiance Fields (NeRF)

Neural Radiance Fields are a recent development introduced by [Mildenhall et al.](https://www.matthewtancik.com/nerf) for 3d image synthesis by optimizing a volumetric function with a multi-layer perceptron network. The achieve impressive results compared to traditional generative techniques.


<!-- 
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1]. -->
<!-- 
## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$. -->

## Reference

[1] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, Angjoo Kanazawa. "Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image." *Proceedings of the ICCV conference on computer vision and pattern recognition*. 2020.

[2] Michael Niemeyer, Andreas Geiger. "GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields" *Proceedings of the CVPR conference*. 2021.

[3] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" *Proceedings of ECCV conference* 2020.

[4] Patrick Esser, Robin Rombach, Björn Ommer. "Taming Transformers For High-Resolution Image Synthesis" *Proceedings of the CVPR conference*. 2021.

[5] Sheng-Yu Wang, David Bau, Jun-Yun Zhu. "Sketch Your Own GANs" 2021.

[6] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, Dahua Lin. "CityNeRF: Building NeRF at City Scale" 2021.

[7] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin. "FastNeRF: High-Fidelity Neural Rendering at 200FPS" *Proceedings of the ICCV conference* 2021.

[8] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever. "DALL-E: Zero-Shot Text-to-Image Generation" 2021.

## Code Repos and Pages

[1] [NeRF](https://github.com/bmild/nerf)

[2] [Infinite Nature](https://github.com/google-research/google-research/tree/master/infinite_nature)

[3] [Giraffe](https://github.com/autonomousvision/giraffe)

