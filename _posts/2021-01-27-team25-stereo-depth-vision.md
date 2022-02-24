---
layout: post
comments: true
title: Depth From Stereo Vision
author: Alex Mikhalev, David Morley
date: 2022-01-27
---


> Our project explores the use of Deep Learning for inferring the depth data based on two side-by-side camera images. This is done by determining which pixels on each image corresponding to the same object (a process known as stereo matching), and then calculating the distance between corresponding pixels, from which the depth can be calculated (with information about the placement of the cameras). While there exist classical vision based solutions to stereo matching, deep learning can produce better results. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

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

[1] Zhuoran Shen, et al. ["Efficient Attention: Attention with Linear Complexities."](https://arxiv.org/pdf/1812.01243v9.pdf) *Winter Conference on Applications of Computer Vision*. 2021.
[2] Vladimir Tankovich, et al. "HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching." *Conference on Computer Vision and Pattern Recognition*. 2021.
[3] Jia-Ren Chang, et al. "Pyramid Stereo Matching Network." *Conference on Computer Vision and Pattern Recognition*. 2018.
[4] Nikolai Smolyanskiy, et al. "On the Importance of Stereo for Accurate Depth Estimation: An Efficient Semi-Supervised Deep Neural Network Approach." *Conference on Computer Vision and Pattern Recognition*. 2018.


---
