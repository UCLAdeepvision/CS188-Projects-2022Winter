---
layout: post
comments: true
title: Text Guided Art Generation
author: Zifan He, Wenjie Mo
date: 2022-01-25
---

> Our project mainly works on investigate and reproduce text-guided image generation model and procedure art creation model (or potentially other image transformer that has artistic values), and connect the two models together to build neuron network that can create artworks with only words. Some artists have already applied AI/Deep Learning in their art creation (VQGAN-CLIP), while the development of diffusion model and transformers may provide more stable and human-like output.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


## Introduction
Our project idea is inspired by [PaintTransformer](https://github.com/wzmsltw/PaintTransformer) which turns a static image into a oil painting drawing process and [VQGAN + CLIP Art generation](https://github.com/nerdyrodent/VQGAN-CLIP) which turns texts into artwork. We want to combine the idea from both project and design a model which could give artistic painting process from text described. Specifically, we want to reproduce and explain VQGAN+CLIP for text guided image generation and take it as the input of the PaintTransformer to produce an artwork with paint strokes. 

To clarify, VQGAN + CLIP is actually the combination of two models: VQGAN stands for *Vector Quantized Generative Adversarial Network*, which is a type of GAN that can be used to generation high-resolution images; while CLIP stands for *Contrastive Language-Image Pretraining*, which is a classifier that could pick the most relevant sentence for the image from several options. Unlike other attention GAN, which can also generate image from text, VQGAN + CLIP is more like a student-teacher pair: VQGAN will generate a image, and CLIP will judge whether this image has any relevance to the prompt and tell VQGAN how to optimize. In this blog, we will focus more on the generative portion and take CLIP as a tool for text-guided art generation process.



## Implementation

## Demo

## Reference
[1] Xu, Tao, et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Liu, Songhua, et al. "Paint transformer: Feed forward neural painting with stroke prediction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[3] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[4] Kim, Gwanghyun, and Jong Chul Ye. "Diffusionclip: Text-guided image manipulation using diffusion models." arXiv preprint arXiv:2110.02711 (2021).

## Code Repository
[1] [VQGAN + CLIP Art generation](https://github.com/nerdyrodent/VQGAN-CLIP)

[2] [AttnGAN](https://github.com/taoxugit/AttnGAN)

[3] [PaintTransformer](https://github.com/wzmsltw/PaintTransformer)

---
