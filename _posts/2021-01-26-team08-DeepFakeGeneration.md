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
GAN ("Generative adversarial network") is the core framework behind most of the deepfake algorithms. The idea is simple, for deepfake generators, the more easily you can trick the human eyes, the better your algorithm is. And for the deepfake detector, the more easily you can detect the fake image, the better. However, we cannot train a deepfake generators by manually evaluating how good the result is-we need the help of the detector. 

That brings the idea of "adversarial": the generator tries to fool the detector, and the detector tries to detect every fake image produced by the generator. And we train these two network at the same time.

![GAN structure]({{ '/assets/images/group08deepfake/GAN-structure.png' | relative_url }})
{: style="width: 800; max-width: 150%;"}
*Fig 1. A simple structure for GAN network. (Image source: <https://neptune.ai/blog/6-gan-architectures>)*

Here we investigate some interesting methods and introduce their ideas through the main innovations in these methods and code explanation:

#### First Order Motion Model for Image Animation

The First Order Motion Model animates an object in the source image according to the motion of a driving video. The method uses self-learned keypoints together with local affine transformations to model complex motions. (Therefore, they call this method a first-order method.) Note that they uses an occlusion-aware generator and extends the equivariance constraint, commonly used for keypoints detection training, to improve the estimation of local affine transformations. <br>

![First-order Model Demo]({{ '/assets/images/group08deepfake/vox-teaser.gif' | relative_url }})
{: style="width: 800; max-width: 150%;"}
*Fig 2. First Order Motion Model on VoxCeleb Dataset. (Image source: <https://github.com/AliaksandrSiarohin/first-order-model/blob/master/sup-mat/vox-teaser.gif>)*


The framework consists of two modules: **the motion estimation module** and **the image generation module**.
The motion estimation module predicts a dense motion field from a frame $$\mathbf{D} \in \mathbb{R}^{3 \times H \times W}$$ of the driving video $$\mathcal{D}$$ to the source frame $$\mathbf{S} \in \mathbb{R}^{3 \times H \times W}$$. The backward optical flow $$T_{S \leftarrow D}$$ is used since backward-warping can be implemented efficiently using bilinear sampling. Also, they assume an abstract reference frame, $$\mathbf{R}$$, which cancels in the later derivations and never explicitly computed, to allow for independent process of $$\mathbf{D}$$ and $$\mathbf{S}$$. 

The motion estimation module is divided to two steps. In the first step, the sparse motion representation is approximated using keypoints and **local affine transformations**. Keypoint locations in $$\mathbf{D}$$ and $$\mathbf{S}$$ are predicted separately by an encoder-decoder network. To model a larger familiy of transformations, they use Taylor expansion to represent $$T_{D \leftarrow R}$$ by a set of keypoint locations and affine transformations. For a given frame $$\mathbf{X}$$, and a set of keypoint coordinates $$p_1, \ldots, p_K$$, the motion function $$T_{X \leftarrow R}$$ is represented by its value in each keypoint $$p_k$$ and its Jacobian computed in each $$p_k$$ location:

$$
T_{X \leftarrow R} \simeq \left\{ \left\{ T_{X \leftarrow R} (p_1), \frac{d}{dp} T_{X \leftarrow R}(p) \bigg\rvert_{p = p_1}\right\}, \ldots, \left\{ T_{X \leftarrow R} (p_K), \frac{d}{dp} T_{X \leftarrow R}(p) \bigg\rvert_{p = p_K}\right\}\right\}
$$

To estimate $$T_{S \leftarrow D}$$ at keypoint $$z_k$$ in $$\mathbf{D}$$ such that $$p_k = T_{R \leftarrow D} (z_k)$$ (so $$z_k = T_{D \leftarrow R} (p_k)$$), we have

$$
\begin{equation*}
\begin{split}
T_{S \leftarrow D} (z_k) &= T_{S \leftarrow R} \circ T_{R \leftarrow D} (z_k) \\
&= T_{S \leftarrow R } \circ T^{-1}_{D \leftarrow R} (z_k)\\
&= T_{S \leftarrow R } \circ T^{-1}_{D \leftarrow R} T_{D \leftarrow R} (p_k) \\
&= T_{S \leftarrow R} (p_k)
\end{split}
\end{equation*}
$$

Using the first order Taylor expansion of $$T_{S \leftarrow R} (p_k)$$ at $$p_k$$, we get 

$$
\begin{equation*}
\begin{split}
T_{S \leftarrow D}(z) &= T_{S \leftarrow R} (p_k)\\
&\approx T_{S \leftarrow R} (p_k) + \left( \frac{d}{dp} T_{S \leftarrow D}(p) \bigg\rvert_{p = p_k} \right)\\
&= T_{S \leftarrow R} (p_k) + \left( \frac{d}{dp} T_{S \leftarrow R} (p) \bigg\rvert_{p = p_k} \right) \left( \frac{d}{dp} T_{D \leftarrow R} (p) \bigg\rvert_{p = p_k} \right)^{-1} (z - T_{D \leftarrow R} (p_k))
\end{split}
\end{equation*} \label{eq1}\tag{1}
$$ 

where $$T_{S \leftarrow R}$$ and $$T_{D \leftarrow R}$$ are predicted by the keypoint predictor (a standard [U-Net](https://en.wikipedia.org/wiki/U-Net) that estimates $$K$$ heatmaps, one for each keypoint). Note that **equivaraince constraints** are imposed when training the keypoint predictor. The deformed images (thin plate spline deformations, specifically) are passed as the inputs to force the model to predict consistent keypoints with respect to known geometric transformations. As shown below, the equivariance loss involves constraints on keypoints as well as on the Jacobians. Suppose an image $$\mathbf{X}$$ is deformed to $$\mathbf{Y}$$ under the transformation $$T_{X \leftarrow Y}$$, the constraints used in this model are 

$$
T_{X \leftarrow R} \equiv T_{X \leftarrow Y} \circ T_{Y \leftarrow R} (p_k) \\
\mathbb{1} \equiv \left(\frac{d}{dp} T_{X \leftarrow R}(p) \bigg\rvert_{p = p_k} \right)^{-1} \left(\frac{d}{dp} T_{X \leftarrow Y}(p) \bigg\rvert_{p = T_{Y \leftarrow R}(p_k)} \right) \left(\frac{d}{dp} T_{Y \leftarrow R}(p) \bigg\rvert_{p = p_k} \right) 
$$

In the second step, a dense motion network combines the local approximations to estimate the dense motion field $$\hat{T}_{S \leftarrow D}$$. The source frame $$\mathbf{S}$$ is warped according to Eq. ($$\ref{eq1}$$) to obtain $$K$$ transformed images $$\mathbf{S^1}, \ldots \mathbf{S^K}$$, so that the inputs are roughly aligned with $$\hat{T}_{S \leftarrow D}$$ when prediction from $$S$$ ($$\hat{T}_{S \leftarrow D}$$ aligns local patterns with pixels in $$\mathbf{D}$$ not $$\mathbf{S}$$). The heatmaps $$\mathbf{H_k}$$, which indicates where each transformation happens, are calculated as below:

$$
\mathbf{H}_z (z) = \exp\left(\frac{(T_{D \leftarrow R} (p_k) - z)^2}{\sigma}\right) - \exp\left(\frac{(T_{S \leftarrow R} (p_k) - z)^2}{\sigma}\right)
$$

Then the $$\mathbf{H_k}$$ and the transformed images $$\mathbf{S}^{1} \ldots \mathbf{S}^{K}$$, as well as the original image $$\mathbf{S}^{0} = \mathbf{S}$$ considered for the background, are passed to a [U-Net](https://en.wikipedia.org/wiki/U-Net), from which $$K+1$$ masks $$\mathbf{M}_k, k = 0, \ldots, K$$ are estimated to indicate where each local transformation holds. Then the final dense motion prediction $$\hat{T}_{S \leftarrow D} (z)$$ is 

$$
\begin{equation}
\begin{split}
\hat{T}_{S \leftarrow D} (z) &= \mathbf{M}_0 z + \displaystyle{\sum_{k = 1}^k \mathbf{M}_k \hat{T}_{S \leftarrow R} (p_k)}\\
&= \mathbf{M}_0 z + \displaystyle{\sum_{k = 1}^k \mathbf{M}_k \left(T_{S \leftarrow R} (p_k) + \left( \frac{d}{dp} T_{S \leftarrow R} (p) \bigg\rvert_{p = p_k} \right) \left( \frac{d}{dp} T_{D \leftarrow R} (p) \bigg\rvert_{p = p_k} \right)^{-1} (z - T_{D \leftarrow R} (p_k))\right)}
\end{split}
\end{equation}
$$

Since image-warping according to the optical flow may not work in the presence of occlusion in $$\mathbf{S}$$, the occlusion map is used to mask out the feature map regions that need to be impainted. Therefore, **the occlusion mask** $$\hat{\mathcal{O}}_{S \leftarrow D}$$ is estimated by adding a channel to the final layer of the dense motion netowrk.

After the the motion estimation module, the generation module is quite straightforward. A generation netowrk is trained to warp the source image according to $$\hat{T}_{S \leftarrow D}$$ an inpaint the image parts that are occluded in the source image. It is worth to notice, however, that the first-order model uses the **relative motion transfer** in the testing stage. Instead of transforming each frame of the driving video $$\mathbf{D}_t$$ to $$\mathbf{S}_t$$ directly in animation of an object in the source frame $$\mathbf{S}_1$$, they transfer the transfer the relative motion between $$\mathbf{D}_1$$ and $$\mathbf{D}_t$$ to $$\mathbf{S}_1$$. For the neighborhood of each keypoint $$p_k$$ of $$D_1$$, $$T_{D_t \leftarrow D_1} (p)$$ is applied:

$$
\displaystyle{T_{S_1 \leftarrow S_t} (z) \approx T_{S_1 \leftarrow R} (p_k) + \left(\frac{d}{dp} T_{D_1 \leftarrow R} (p) \bigg\rvert_{p = p_k} \right) \left(\frac{d}{dp} T_{D_t \leftarrow R} (p) \bigg\rvert_{p = p_k} \right)^{-1} (z - T_{S \leftarrow R} (p_k) + T_{D_1 \leftarrow R}(p_k) - T_{D_t \leftarrow R}(p_k))}
$$

This equation can actually be derived from Eq. ($$\ref{eq1}$$). Note that one assumption of the first-order model is that $$\mathbf{S_1}$$ and $$\mathbf{D_1}$$ have similar poses. Otherwise, the idea of relative motion transfer here might not work.



### Relevant papers and their git repo:

1. First Order Motion Model for Image Animation<br>
    [GitHub Link](https://github.com/AliaksandrSiarohin/first-order-model)

2. StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation<br>
    [GitHub Link](https://github.com/yunjey/StarGAN/)

3. StarGAN v2: Diverse Image Synthesis for Multiple Domains<br>
    [GitHub Link](https://github.com/clovaai/stargan-v2/)

4. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks<br>
    [GitHub Link](https://github.com/junyanz/CycleGAN/)

5. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks<br>
    [GitHub Link](https://github.com/xinntao/ESRGAN/)

6. Image-to-Image Translation with Conditional Adversarial Networks<br>
    [GitHub Link](https://github.com/phillipi/pix2pix/)

7. MaskGAN: Towards Diverse and Interactive Facial Image Manipulation<br>
	[GitHub Link](https://github.com/switchablenorms/CelebAMask-HQ/)

8. DeepFaceLab: Integrated, flexible and extensible face-swapping framework<br>
    [GitHub Link](https://github.com/iperov/DeepFaceLab/)

9. FSGAN: Subject Agnostic Face Swapping and Reenactment<br>
    [GitHub Link](https://github.com/YuvalNirkin/fsgan/)



## Reference