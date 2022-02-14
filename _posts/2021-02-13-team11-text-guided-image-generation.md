---
layout: post
comments: true
title: Text Guided Image Generation
author: Devin Yerasi, Jing Zou
date: 2022-01-18
---


> Text-guided image generation is an important milestone for both natural language procesing and computer vision. It seeks to use natural language prompts to generate new images or edit previous images. Recently diffusion models have been shown to produce better results than GANS in regards to text-guided image generation. In this article, we will be examining GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## Introduction


## Architecture

## Implementation

## Demo

<!--Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)-->

<!--
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
-->
## Reference

[1] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I. and Chen, M., 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741.

[2] Dhariwal, P. and Nichol, A. Diffusion models beat gans on
image synthesis. arXiv:2105.05233, 2021.

[3] Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. arXiv:2006.11239, 2020.

[4] Ho, J. and Salimans, T. Classifier-free diffusion guidance.
In NeurIPS 2021 Workshop on Deep Generative Models
and Downstream Applications, 2021. URL https://
openreview.net/forum?id=qw8AKxfYbI.

[5] Nichol, A. and Dhariwal, P. Improved denoising diffusion
probabilistic models. arXiv:2102.09672, 2021.

---
