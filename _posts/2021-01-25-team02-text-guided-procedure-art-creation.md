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

## Technical Details of Models

### AutoEncoder

**Autoencoder** is a type of neuron network frequently used to learn the representations of images. Similar to what we have in Fast R-CNN model in object detection, it has an encoder that maps the input image to a representation tensor normally named as *latent vector*, shown as the yellow box in Fig 1, and a decoder that upsample the latent vector, or more intuitively, revert the encoding to reconstruct the original image. If we can find an encoder-decoder combination that can minimize the reconstruction loss, then we can compress the image into the latent vector for other operations without significant loss so that we can use resources more efficient. In Fast R-CNN, we do ROI proposal over the code instead of the original image. This is an unsupervised learning technique, meaning that we don't use the labels of images to compute the loss.

![Autoencoder]({{ '/assets/images/team02/Autoencoder_schema.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. High level architecture of an autoencoder* [5].

For encoder, the structure would be the same as any normal neuron network we used for image classification without the last few fully-connected layer (otherwise we will lose too much information), including MLP (only linear layers), CNN, and ResNet. For decoder, we can use a combination of max unpooling, linear layers with increasing inputs, or transposed convolutional layer if the encoder is constructed with CNN or ResNet for upsampling. A simple example of an autoencoder in Pytorch is shown below:

```python
class AutoEncoderConv(torch.nn.Module):
  def __init__(self):
    super().__init__()

    # Encoder
    self.encoder_conv1 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 3, 1, 1),
        torch.nn.ReLU()
    )

    self.encoder_conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(16, 32, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, 1, 1),
        torch.nn.ReLU()
    )

    self.encoder_conv3 = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, 1, 1),
        torch.nn.ReLU()
    )

    # Decoder
    self.decoder_conv1 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(64, 64, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 32, 2, 2, 0),
        torch.nn.ReLU()
    )

    self.decoder_conv2 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(32, 32, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(32, 16, 2, 2, 0),
        torch.nn.ReLU()
    )

    self.decoder_conv3 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(16, 16, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(16, 3, 2, 2, 0),
        torch.nn.Sigmoid()
    )

  def forward(self, x):
    x = self.encoder_conv1(x)
    x = F.max_pool2d(x, 2)
    x = self.encoder_conv2(x)
    x = F.max_pool2d(x, 2)
    x = self.encoder_conv3(x)
    code = F.max_pool2d(x, 2)
    x = self.decoder_conv1(code)
    x = self.decoder_conv2(x)
    x = self.decoder_conv3(x)
    return x
```
We can optimize the model based on the return value `x`. Training this model over CIFAR-10, we can get an 0.56 average training loss in 15 epochs. However, there is a main drawback: since it is optimizing the mapping between input image and the latent vector, it works well for training images but worse for others. To have a broader usage setting, we need the **variational autoencoder**(VAE).

### VQVAE

Differ from the standard autoencoder, **variational autoencoder** tries to find a probability distribution of the latent vector instead of a mapping. VAEs also have encoders and decoders like autoencoders, but since the latent space is now a continuous distribution, we also need to optimize the parameters of this distribution. The idea was firstly introduced by Diederik P. Kingma and Max Welling [7] under the context of Bayesian estimation: assume the input data $$\mathbf{x}$$ has an unknown distribution $$P(x)$$. We know guess that it is a normal distribution with parameter group $$\theta$$ (it can be a mean and a variance, or multiple of them to get a mixture of normal distribution). We start from $$\mathbf{z}$$ as a initial guess of the parameters (usually called the *latent encoding*). Let $$p_{\theta}(\mathbf{x}, \mathbf{z})$$ be the joint probability. The marginal probability of $$\mathbf{x}$$ is

$$
p_\theta (\mathbf{x}) = \int_{\mathbf{z}} p_{\theta}(\mathbf{x}, \mathbf{z}) d\mathbf{z} = \int_{\mathbf{z}} p_{\theta}(\mathbf{x}|\mathbf{z})p_\theta(\mathbf{z}) d\mathbf{z}
$$

By the Bayesian theorem,

$$
p_\theta (\mathbf{z} | \mathbf{x}) = \frac{p_\theta (\mathbf{x}|\mathbf{z}) p_\theta (\mathbf{z})}{p_\theta(\mathbf{x})}
$$

In every iteration, we can compute this posterior and use it as the prior for the next iteration. Since we don't know the real $$\theta$$, we just use our own estimate *codebook* $$\Theta$$ and optimize it gradually.

The original VAE uses normal distribution as the proposal of underlying phenomenon, but in **Vector-Quantized VAE** (VQVAE), the parameters are several tensors (embedding) with fixed length called *Codebook* and uses a data compression method "vector quantization" to generate the latent encoding. What this method basically do is to find the embedding vector with the lowest distance to the flattened tensors before quantization, store the corresponding index, and replace them with those embedding vectors. Mathematically, the quantized latent vector has the formula

$$
z_q = argmin_{z_k \in Z} \| z_{ij} - z_k \|
$$

where $$Z$$ is the codebook, $$z_{ij}$$ is a flatten tensor after encoder, and $$z_q$$ is the quantized latent vector [3].

Concerning calculation of loss, there are three types of losses: reconstruction loss, which basically calculate the distance between reconstructed image and original image; stop-gradient loss for embedding, which optimize the embedding network; and commitment loss, which optimize the encoder. The last two loss is specifically for the vector quantization. The overall loss is the sum of three, which is

$$
\mathcal{L}_{VAE}(E,D,Z) = \|x - \hat{x}\|^2 + \|sg[E(x)]-z_q\|^2 + \|E(x) - sg[z_q]\|^2
$$

where $$D$$ is the decoder, $$\hat{x}$$ is the reconstructed image, and $$sg[\cdot]$$ stands for stop gradient.

### GPT-2

### VQGAN

### VQGAN + CLIP

### PaintTransformer

## Implementation

## Demo

## Reference
[1] Xu, Tao, et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Liu, Songhua, et al. "Paint transformer: Feed forward neural painting with stroke prediction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[3] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[4] Kim, Gwanghyun, and Jong Chul Ye. "Diffusionclip: Text-guided image manipulation using diffusion models." arXiv preprint arXiv:2110.02711 (2021).

[5] Michela Massi - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=80177333

[6] https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

[7] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

## Code Repository
[1] [VQGAN + CLIP Art generation](https://github.com/nerdyrodent/VQGAN-CLIP)

[2] [AttnGAN](https://github.com/taoxugit/AttnGAN)

[3] [PaintTransformer](https://github.com/wzmsltw/PaintTransformer)

---
