---
layout: post
comments: true
title: Deepfake Portraits using Style Transfer GANs
author: Sabrina Liu, Daniel Smith
date: 2022-01-27
---


> We will investigate generation of deepfake portraits using style-based generative adversarial network models. In particular, we will examine [StyleGAN](https://github.com/NVlabs/stylegan), its successor [StyleGAN2](https://github.com/NVlabs/stylegan2), and [StarGAN2](https://github.com/clovaai/stargan-v2).

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Within the realm of generating pictues of fake people, there are several different avenues of research. The first few generative models were generative adversarial network models in which a generator and discriminator would compete against one another to improve deepfake generation and detection, respectively. Later, research extended to using GANs for style-based transfer to generate new faces, which involves extracting both coarse styles like pose, facial expression, and lighting as well as fine styles like hair, and applying those styles to a source image. In this article, we will examine several state-of-the-art style-transfer GANs which are able to produce high-resolution, hyperrealistic images of fake people.


## Reference
[1] Tero Karras, Samuli Laine, and Timo Aila, “[A style-based
generator architecture for generative adversarial networks](https://arxiv.org/pdf/1812.04948.pdf),” in
CVPR, 2019.

[2] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten,
Jaakko Lehtinen, and Timo Aila, “[Analyzing and improving
the image quality of stylegan](https://arxiv.org/pdf/1912.04958.pdf),” in CVPR, 2020.

[3] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha,
Sunghun Kim, and Jaegul Choo, “[StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/pdf/1912.01865.pdf),” in CVPR, 2020.

<!-- 
## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



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

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

--- -->
