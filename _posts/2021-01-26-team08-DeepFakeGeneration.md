---
layout: post
comments: true
title: DeepFake Generation
author: Zhengtong Liu, Chenda Duan
date: 2022-01-26
---

> This is the blog will record and explain technical details for Zhengtong Liu and Chenda Duan's CS188 DLCV project.
> We will investigate the state-of-the-art DeekFake Generation methods. We are not sure yet which specific subtract we will focus on.
> Here are the potential fields: Image Synthesis, Image Manipulation, Face Swapping.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
## Team08 Initial Project Proposal

### Group  Members: Chenda Duan, Zhengtong Liu

### Topic: DeepFake Generation (with emphasis on image to image translation using GAN)

## What is deepfake?
What is [deepfake](https://en.wikipedia.org/wiki/Deepfake/)? It is a newly emerged term created by some reddit users. In short, it refers to using [deep learning](https://en.wikipedia.org/wiki/Deep_learning) to generate "fake" images, which look like photos captured in the real world but is not.

The most heated use of deep fake is to do the face swapping, whichis the subset of a broader definition called "deepfake generation".  As you might see in recent years, more and more people get access to well-performed deep fake algorithms and create funny, weird images that might even cause problems.

Here is an example of using deepfake generation ([imgsrc](https://news.artnet.com/art-world/mona-lisa-deepfake-video-1561600/)) 
![deepfake example]({{ '/assets/images/group08deepfake/Mona-Lisa-deepfake.png' | relative_url }})
{: style="width: 800; max-width: 150%;"}
*Fig 1. A example of deepfake generation. (Image source: <https://news.artnet.com/art-world/mona-lisa-deepfake-video-1561600>)*

As deepfake generation might cause many problems (such as fake news!), the other popular subtract is deepfake detection, where we try to build the network that can identify real images from fake, generated images.

## Core ideas: GAN
GAN ("Generative adversarial network")

### Relevant papers and their git repo:

1. MaskGAN: Towards Diverse and Interactive Facial Image Manipulation<br>
	GitHub Link: https://github.com/switchablenorms/CelebAMask-HQ

2. StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation<br>
    GitHub Link: https://github.com/yunjey/StarGAN

3. StarGAN v2: Diverse Image Synthesis for Multiple Domains<br>
    GitHub Link: https://github.com/clovaai/stargan-v2

4. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks<br>
    GitHub Link: https://github.com/junyanz/CycleGAN

5. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks<br>
    GitHub Link: https://github.com/xinntao/ESRGAN

6. Image-to-Image Translation with Conditional Adversarial Networks<br>
    GitHub Link: https://github.com/phillipi/pix2pix

7. DeepFaceLab: Integrated, flexible and extensible face-swapping framework<br>
    GitHub Link: https://github.com/iperov/DeepFaceLab

8. FSGAN: Subject Agnostic Face Swapping and Reenactment<br>
    GitHub Link: https://github.com/YuvalNirkin/fsgan

## Reference