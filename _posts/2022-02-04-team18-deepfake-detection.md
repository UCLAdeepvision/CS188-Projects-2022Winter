---
layout: post
comments: true
title: Proposal

author: UCLAdeepvision
date: 2022-02-04
---

> With the rise of more sophistacated methods to fabricate virtual content, it has become imperative to develop techniques for the detection of artificially generated media. This project explores the identification of edited or fabricated images using Generative Adversarial Networks.  


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## What is a Deepfake?
Deepfakes are artificially generated images and videos that are difficult to distingish from authentic content. These images and videos can have massive social and political implications, which has led to the development of sophisticated techniques to detect them. Nowadays the most common approach to produce deepfakes is using deep learning models such as autoencoders and Generative Adversarial Networks. Detecting deepfakes is a binary classification problem: images are categorized as fake or real. While early approaches to this problem involved hand-tuning features, newer deep-learning approaches make use of CNNs and GANs to produce better results. These methods allow for better and more fine-tuned feature extraction, which in turn allows for more nuanced detection.  

## Detecting Deepfakes: Generative Adversarial Networks 

While Deepfake generation has become a standard problem that is commonly solved with the use of GANs, detection is a much more nuanced task. One option is  the architecture now known as MesoNet, which is a CNN architecture. The architecture consists of four alternating layers of convolutions and pooling, a dense network, and hidden layer and utilizes Batch normalization and ReLu activation. The other solution path involves utilizing the fact that deep fakes are generated with GANs. By extracting the discriminators, we can use them as a module specifically to detect deep fakes.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.

Project Repository:
https://github.com/arnavgarg/UCLA_CS_188_Final_Project

References:

https://arxiv.org/abs/2101.09781

https://ieeexplore.ieee.org/document/9564096

https://www.pnas.org/content/119/1/e2110013119

Project page: https://ucladeepvision.github.io/CS188-Projects-2022Winter/

