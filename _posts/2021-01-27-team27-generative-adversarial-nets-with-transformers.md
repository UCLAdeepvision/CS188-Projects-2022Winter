---
layout: post
comments: true
title: Generative Adversarial Networks with Transformers
author: Sidi Lu
date: 2022-01-27
---


> While Vision Transformers have caught quite some attention in the community, it is still yet to be explored how such powerful models could work on building powerful GANs. Based on some recent progress in studying Transformers' position encoding system, we want to explore the possibility of building a vision-oriented transformer block that is simple, light-weighted yet effective for a stable training of transformer GANs.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
## Introduction
There's already one successful attempt at training transformer GANs ([TransGAN](https://arxiv.org/pdf/2102.07074.pdf)). However, as indicated by the paper, the proposed approach is showing the most significant shortcoming of almost all transformer-based models - it is more data-hungry compared to other architectures by a large margin. Inspired by the recent finding of an un-embedded, prior-based position encoding system ([LinearPos](https://arxiv.org/abs/2108.12409)), we hereby motivate our exploration of a simple, light-weighted yet effective design of a vision-oriented transformer block. We hope such design could alleviate the data efficiency problem of transformers whereas to maintain the merit of its ability to model long dependencies.

## Implementation
TBD.

The 1-D Linear Bias is now extended to the 2-D world. Multiple diffusion pattern should examined, such as linear, gaussian etc.
## Demo
TBD

## Reference
[1] Jiang Y, Chang S, Wang Z. Transgan: Two transformers can make one strong gan[J]. arXiv preprint arXiv:2102.07074, 2021, 1(2): 7.

[2] Press O, Smith N A, Lewis M. Train short, test long: Attention with linear biases enables input length extrapolation[J]. arXiv preprint arXiv:2108.12409, 2021.

[3] Durall R, Frolov S, Hees J, et al. Combining transformer generators with convolutional discriminators[C]//German Conference on Artificial Intelligence (Künstliche Intelligenz). Springer, Cham, 2021: 67-79.

[4] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks[C]//International conference on machine learning. PMLR, 2017: 214-223.

[5] Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of wasserstein gans[J]. arXiv preprint arXiv:1704.00028, 2017.

## Code Repository
[1] [WGAN/WGAN-GP](https://github.com/Zeleni9/pytorch-wgan.git)

[2] [TransGAN](https://github.com/VITA-Group/TransGAN)

[3] [WGAN-T (Ours)](https://github.com/desire2020/WGAN-T)

---
