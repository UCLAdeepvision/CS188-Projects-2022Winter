---
layout: post
comments: true
title: MMDetection
author: Yu Zhou, Zongyang Yue
date: 2022-01-27
---


> In this paper, we study state of the art object detection algorithms and their implementations in MMDetection. We measure their performances on main-stream benchmarks such as the COCO dataset, and further evaluate their performances against adversarial attacks.
We explore and try to understand the library by removing parts of it and checking the change in performances of our algorithms.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content


<!-- Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md) -->

<!-- ## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work. -->

<!-- 
### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          | -->



<!-- ### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
``` -->


<!-- ### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/). -->

## Reference

[1] Liu, Ze, et al. "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[2] Wang, Wenhai, et al. "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction Without Convolutions." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[3] He et al. "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016.
[4] Wang et al. "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 2021.
[5] Wang et al. "Pvtv2: Improved baselines with pyramid vision transformer." *arXiv preprint arXiv:2106.13797*. 2021.
[6] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." *arXiv preprint arXiv:2004.08955*. 2021.
[7] Gao, Shang-Hua, et al. "Res2Net: A New Multi-Scale Backbone Architecture." *Proceedings of the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*. 2021.
[8] Chen, Kai, et al. "MMDetection: Open MMLab Detection Toolbox and Benchmark. " *arXiv preprint arXiv:1906.07155*. 2019.
---



## Code Repository

[MMDetection Main](https://github.com/open-mmlab/mmdetection)

[PVT and PVTv2](https://github.com/whai362/PVT)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[ResNeSt](https://github.com/zhanghang1989/ResNeSt)

[Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels)

[ResNet](https://github.com/KaimingHe/deep-residual-networks)