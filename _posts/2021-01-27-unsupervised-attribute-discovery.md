---
layout: post
comments: true
title: Unsupervised Attribute Discovery to Generate 3D models
author: Team 4 (Amanda Han and Timothy Kanarsky)
date: 2022-01-26
---


> Our goal is to utilize semantic factorization of GANs in order to generate unseen renderable 3D models with neural rendering. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Most relevant papers:
1. SeFa: [Closed-Form Factorization of Latent Semantics in GANs](https://genforce.github.io/sefa/#demo)
2. [Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering](https://nv-tlabs.github.io/GANverse3D/)
3. [3D-aware Image Synthesis via Learning Structural and Textural Representations](https://genforce.github.io/volumegan/)

## Basic Syntax
### Image

![Anime]({{ '/assets/images/team04/animeImage.gif' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Anime style factorization* [1].



Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

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

[1] Shen, Yujun, and Bolei Zhou. “Closed-Form Factorization of Latent Semantics in GANs.” Sefa, https://genforce.github.io/sefa/#demo. 

---
