---
layout: post
comments: true
title: Anime Sketch Colorization and Style Transfer
author: Jeremy Tsai
date: 2022-02-23
---


> In this project I explore how to color a sketch in the style according to another colored image.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
This project is based on the paper "Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN" by Zhang et al. 

## Theory
The U-Net is initially proposed by Ronneberger et al. in 2015 for biomedical Image Segmentation [1]. In a segmentation task, a class label should be assigned to each pixel, so it is important for localization. In a normal convolutional network, the later layers have a receptive field that essentially encompasses the whole image, which goes against the idea of "localization". As such, in addition to the normal contracting layers of convolutional neural networks, Ronneberger et al. proposed appending an expanding branch to the normal contracting layers of convolutional networks. Furthermore, there are "skip connected layers", where feature maps from earlier layers skip over layers and gets appended to the expanding branch. This is clearly illustrated in Figure 1.

![U-Net Outline]({{ '/assets/images/team21/u_net_architecture.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}This is the model architecture as proposed in the original U-Net paper. [1]

Zhang et al. built on top of the U-Net for this model, as coloring can be thought of as a classification task per pixel as well. The exact model is illustrated in Figure 2. The exact changes are the addition of the pre-trained VGG-19 models and the Guide Decoders. The reasoning behind the VGG layer is that it is meant to produce the "style vector". The idea is that at the end of the contracting branch of the U-Net, we would have a vector that is a high-level, global representation of the sketch. By adding the VGG style vector, we would be adding color into this representation, and so when we go through the expanding branch, we would reconstruct the original sketch with the colors added.

![Zhang et al.]({{ '/assets/images/team21/zhang_net.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}This is the model architecture in the Zhang paper. [2]

Another addition to the U-Net that Zhang et al. made is the addition of guide decoders. These are added to prevent the vanishing gradient problem. They are added at the entry and exit of mid-level layers, and produce a monochrome and colored image each. These images are used to calculate the loss in the generator. The precise loss is given as:

$$
\mathbb{E}_{x,y\sim P_{data}(x,y)}[\lVert y-G_f(x,V(x) \rVert_1+\alpha\lVert y-G_{g_1}(x)\rVert_1+\beta\lVert T(y)-G_{g_2}(x,V(x))\rVert_1]
$$
where $x$ is the input sketch, $V(x)$ is the output of the VGG 19 layer (refer to the Architecture section below), $G_f(x,V(x))$ is the model's output colorized image, $G_{g_1}(x)$ and $G_{g_2}(x)$ are the outputs of the two guide decoders, $y$ is the label colorized image, $T(y)$ is the grayscaled label colorized image. $\alpha$ and $\beta$ are magic constants that are 0.3 and 0.9, respectively. These numbers are tested to be optimal empirically by Zhang et al. [1]. Note that the contribution to the loss function by the first guide decoder is the difference between its output and the grayscaled colorized image. This is because the first guide decoder is placed at the entry of the mid-level layer, before the addition of the color. Naturally, the reconstructed image would thus have no color.

Zhang et al. has implemented a Discriminator that would be optimized along with the U-Net, but this part has not been explored (but will be by the end of this project).

## Dataset
The original Zhang et al. paper did not mention the exact dataset they trained the dataset by. However, this project is continued to be explored by Zhang and his group of researchers [3]. In the paper that outlined the third-generation of this model, they mentioned training using data scraped from the Danbooru database [4]. The Danbooru database is a large-scale anime anime image database with user-annotated tags [6]. Although this precise paper outlined a model architecture that is significantly different (it utilizes two U-Nets instead of just one), due to the similarity of the task, the same training data would be used. As such, for this project, we scraped all images that satisfies the following criteria: (1) it is part of the Danbooru 2021 512 by 512 pixels dataset, (2) it has a "s", meaning Safe-For-Work, rating, and (3) it has the tag `white_background`. The `white_background` tag describes images that have a colorized character, but with a white background. The need for a blank background is because the model is not designed to color arbitrary anime characters in any settings. Zhang et al. posited that there are other model architectures that are more suitable to colorize manga or images with background [3]. After all of these images are scraped, they are all resized to 256 by 256. When they are fed into the model, the pixel values are normalized from varying between 0 to 255 to between 0 to 1.

 Using this method and help from a script written by user Atom-101 on Github, a total of 31187 images are scraped [6]. These scraped images are colored images, and so they serve as our "colorized" images (the labels). The database doesn't actually have the sketch version of these images, so these have to be extracted from these colorized images. The sketch images (part of the input) are extracted from the original image by applying a grayscale, an inversion, and a Gaussian blur. Zhang et al. specified that their dataset is extracted using a neural network trained by Zhang et al. [4]. However, due to the slow speed of the model (it takes two second to process an image), this option was abandoned.
 
 Now that we have obtained the colored image and the sketch, we still need the style image input. However, the Danbooru database does not readily satisfy this need, as it does not have multiple colorized version of a colorized sketch, each with a different style. Furthermore, there is no objectively "right answer" to what a "sketch that is colorized according to a particular style" is. As such, we notice that the colorized images that we are using as the labels are just the sketches colored to the style of itself. As such, the original scraped images that we are using as the label are also used as tho input to serve as the style image.

## Architecture Implementation and Training
The model architecture is implemented as closely as possible to the original paper. However, due to the paper not having the clearest illustration and verbal descriptions, some parts have to be inferred and this might have caused some discrepancy between this project's implementation and the original implementation. These inferences (including placement of ReLU layers, pooling methods, padding methods, and depths of skipped connected layers) are those that are described in the original U-Net [1,2]. Efforts are made to salvage the original model by scouring the GitHub repositories by Zhang, but the original source code used to train the model had never been published, and the pre-trained model also seemed to be lost since 2018 [3]. The PyTorch implementation is as outlined below:

```
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.downsample_128 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 16 * 256 * 256
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 16 * 256 * 256
        nn.MaxPool2d(kernel_size=2), #16 * 128 * 128
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 32 * 128 * 128
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, padding_mode='reflect') # 32 * 128 * 128
    )
    self.downsample_64 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2), #32 * 64 * 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 64 * 64 * 64
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 64 * 64 * 64
    )
    self.downsample_32 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2), #64 * 32 * 32
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 128 * 32 * 32
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 128 * 32 * 32
    )
    self.downsample_16 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2), # 128 * 16 * 16
        nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 256 * 16 * 16
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 256 * 16 * 16
    )
    self.bottom_8 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2), # 256 * 8 * 8
        nn.Conv2d(256, 2048, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 2048 * 8 * 8
    )
    self.vgg = VGG_19_fc_1_output()
    self.vgg_2048 = nn.Linear(4096, 2048)
    self.upsample_16_1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2) # 512 * 16 * 16
    self.upsample_16_2 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 512 * 16 * 16
    )
    self.upsample_32 = nn.Sequential(
        nn.ConvTranspose2d(512 + 256, 128, kernel_size=2, stride=2), # 128 * 32 * 32
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 128 * 32 * 32
    )
    self.upsample_64 = nn.Sequential(
        nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2), # 64 * 64 * 64
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 64 * 64 * 64
    )
    self.upsample_128 = nn.Sequential(
        nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2), # 32 * 128 * 128
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 32 * 128 * 128
    )
    self.upsample_256 = nn.Sequential(
        nn.ConvTranspose2d(32 + 32, 64, kernel_size=2, stride=2), # 64 * 256 * 256
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 64 * 256 * 256
        nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1) # 3 * 256 * 256
    )
    # Guide decoders
    self.guide_dec_1 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 512 * 16 * 16
        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # 256 * 32 * 32
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 128 * 64 * 64
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 64 * 128 * 128
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 32 * 256 * 256
        nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1) # 3 * 256 * 256
    )
    self.guide_dec_2 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, padding_mode='reflect'), # 512 * 16 * 16
        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # 256 * 32 * 32
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 128 * 64 * 64
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 64 * 128 * 128
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 32 * 256 * 256
        nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1) # 3 * 256 * 256
    )
    
  def forward(self,x):
    sketch, style = x
    skip_128 = self.downsample_128(sketch)
    skip_64 = self.downsample_64(skip_128)
    skip_32 = self.downsample_32(skip_64)
    skip_16 = self.downsample_16(skip_32)
    self.vgg.eval()
    with torch.no_grad():
      vgg_4096 = self.vgg(style)
    vgg_hint = self.vgg_2048(vgg_4096).unsqueeze(2).unsqueeze(3)
    bottom_8 = self.bottom_8(skip_16)
    bottom_sum = vgg_hint + bottom_8
    up_16_1 = self.upsample_16_1(bottom_sum)
    up_16 = self.upsample_16_2(up_16_1)
    up_32 = self.upsample_32(torch.cat([up_16, skip_16], dim=1))
    up_64 = self.upsample_64(torch.cat([up_32, skip_32], dim=1))
    up_128 = self.upsample_128(torch.cat([up_64, skip_64], dim=1))
    # turn decoder 1 result into grayscale
    grayscale_dec_1 = torch.mean(self.guide_dec_1(skip_16), dim=1).unsqueeze(dim=1)
    return grayscale_dec_1, self.guide_dec_2(up_16_1), self.upsample_256(torch.cat([up_128, skip_128], dim=1))
```
Note that the `forward()` function returns the output of the guide decoders and the actual output image. This is for convenience of calculating the loss function. The `VGG_19_fc_1_output()` is the pre-trained VGG-19 model, except given an image, it produces a vector of size 4096 (the input to the first fully connected layer). This is in accordance to the design outlined by Zhang et al. in their paper [2].

## Preliminary Results
The model is trained using the Adam optimizer using default PyTorch parameters (`lr=0.001`, `betas=(0.9, 0.999)`, `eps=1e-08`). 70% of the 31189 scraped images is used as the training set, 20% as the validation set, and 10% as the test set. The batch size is 32 images, and the model has trained for around 3000 iterations (about 5 epoch). Note that the loss function used for now is not the same as the actual paper, but rather a single term in the two-term summation that forms the actual loss function. More research is needed to understand the significance of the second term-summation. The preliminary results of the model is clear in the example below, where we fed it with a sketch from the test set paired with two different style images from the test set:

![Preliminary Results]({{ '/assets/images/team21/prelim_results.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}The image drawn by the model after 5 epoch. [1]

The model at this point can correctly recreate the sketch after the U-Net to a fairly high degree of accuracy, while giving it some sort of color. However, these color are more of a purplish-gray shade rather than actual coloring. It is clear that even when the style image is the colored version of the sketch itself, the color of the output image at this point bares absolutely no semblance to the original colored version. Furthermore, when given an arbitrary style image, the output image is nearly identical to when the style image is the colored sketch. It remains to be seen whether more training epoch would fix this problem, or the addition of the discriminator (i.e. the incorporation of the AC-GAN) would solve this issue.

Another issue that is shown by the output right now is the presence of the random colored dots that are particularly rampant at the sides of the images. This is presumably a side effect of the black sides of the images from the Danbooru dataset, so the training set might need to be refined.

## Reference

[1] Lvmin Zhang, Yi Ji, Xin Lin. 2017. Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN. arXiv:1706.03319
[2] Olaf Ronneberger, Philipp Fischer, Thomas Brox. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv:1505.04597
[3] The Style2PaintsResearch Team, Style2Paints, (2018), GitHub repository, https://github.com/lllyasviel/style2paints
[4] Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. 2018. Two-stage sketch colorization. ACM Trans. Graph. 37, 6, Article 261 (December 2018), 14 pages. DOI:https://doi.org/10.1145/3272127.3275090
[5] Atom-101, Danbooru Dataset Maker, (2020), GitHub repository, https://github.com/Atom-101/Danbooru-Dataset-Maker
[6] Anonymous, The Danbooru Community, & Gwern Branwen; “Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset”, 2022-01-21. Web. Accessed February 20, 2022 https://www.gwern.net/Danbooru2021

---
