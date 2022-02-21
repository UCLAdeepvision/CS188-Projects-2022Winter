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
\mathcal{L}_{VAE}(E,G,Z) = \|x - \hat{x}\|^2 + \|sg[E(x)]-z_q\|^2 + \|E(x) - sg[z_q]\|^2
$$

where $$G$$ is the decoder(generator), $$\hat{x}$$ is the reconstructed image, and $$sg[\cdot]$$ stands for stop gradient.

### Transformer

So far we have the model to learn local interactions between image pixels. The original paper for VQGAN [3] also mentioned that long-range interactions are also important for high-resolution image synthesis and image recovery. We would like to know what pixel would be next given all pixels we perceived previously. This idea is similar to the next-word prediction in NLP using transformer models. A revolutionary idea in VQGAN architecture is using a partial GPT-2 transformer model to train the long-range interaction.

Transformer is a model mainly based on the self-attention mechanism in NLP, which the model would put more weight (attention) on certain portion, similar to how human perceive and understand a sentence. Fig 2 is a diagram of the architecture of a transformer.

![Transformer]({{ '/assets/images/team02/transformer.svg' | relative_url }})
{: style="max-width: 90%; align-items: center"}
*Fig 2. The transformer architecture (VQGAN only uses the decoder part since vector quantization has already produce the encoding)* [8].

For VQGAN, only the decoder will be used, since vector quantization in VQVAE has already generated both latent vector and the encoding at the same time. The encoding is a long discrete sequence of numbers, and the goal of transformer is to predict each number in the sequence based on all previous numbers with the maximum likelihood. Specifically, we want to find

$$
argmax_p [\Pi_i p(s_i|s_{<i})]
$$

Where $$s_i$$ is the ith number in the sequence [9]. Therefore, the loss function of this transformer would be $$\mathbb{E}(-\log{(\Pi_{i} p(s_i\vert s_{<i}))})$$, which is the negative-log loss.

### VQGAN

Combining all models above together, we can finally build the **VQGAN** with the following architecture:

![VQGAN]({{ '/assets/images/team02/vqgan.png' | relative_url }})
{: style="width: 100%; max-width: 100%;"}
*Fig 3. Overall VQGAN Architecture* [8].

We still miss a component: the discriminator. In GAN, during training, there are usually one generator to construct the target image, and a discriminator, which is nothing but a classifier, that is used to distinguish whether the image is real or fake and compute the loss. Both generator and discriminator optimize and compete with each others, so that we can get better performance on the generator. In VQGAN, a convolutional discriminator will take the image generated by VQVAE (the generator) and output a real-fake matrix, which is a square matrix with 0's and 1's. For the project, since we are training on CIFAR-10, where each image only have $$32\times 32$$ dimension, it still works well even we only output a single true-false bit from the discriminator. 

For the loss function, we use the minmax loss, which is commonly used for GAN:

$$
L_{GAN} = \log{D(x)} + \log{(1-D(\hat{x}))}
$$

where $$D$$ is the discriminator, and $$\hat{x}$$ is the reconstructed image. When training the whole VQGAN without the transformer, we need to add the loss from VQVAE and this loss with some adaptive coefficient. In our implementation, we have 7 epoch warm up stage that the adversarial loss will not contribute in the optimization, then apply the loss with coefficient 0.001, since discriminator may not perform well at the beginning, and introduce loss too early may cause over confidence of the generator.

Lastly, for the training procedure, we first train the VQVAE with discriminator to obtain the best latent codebook that can represent the distribution of the image. After that, we train the transformer based on the codebook and encoder to model the long-range interaction. When generating the image, the original method is having a sliding window of fixed size over a region and only predict pixels with the neighbors in the window. This would allow parallelism and save more resources, but since we would only generate a $$32\times 32$$ as the demo, it is reasonable to directly predict over the whole image.

### VQGAN + CLIP

To incorporate VQGAN with CLIP to provide the power of image generation from text, we will download the pretrained CLIP model from OpenAI (reimplement it would time consuming and may not obtain the same performance). For VQGAN, we only need the decoder and the codebook. Our goal is trying to find a latent vector before vector quantization that can produce a figure corresponding to the text prompt. Initially, we randomly generate a latent vector with min and max bounded by the minimum and maximum in the codebook. Then the pretrained CLIP model will parse the prompt into tokens and encode them. It also output the weight of token and construct a Prompt object for loss calculation. Each Prompt object behaves like a self-attention unit that computes the distance between CLIP prediction on the generated image and the embedding and scales by a weight. 

During training, the system will first quantize the latent vector, decode it to an normalized image, perform an augmentation, and output an encoding by the CLIP model. We then put the output into each Prompts to calculate the loss. Finally, based on the loss, we optimize the latent vector and repeat the operation. A workflow is shown below:

![VQGAN]({{ '/assets/images/team02/vqgan+clip.svg' | relative_url }})
{: style="width: 100%; max-width: 100%;"}
*Fig 4. Workflow of VQGAN+CLIP text-guided image generation*.

### PaintTransformer

## Implementation

### VQGAN + CLIP

All models for this system are trained using CIFAR-10, where each image has shape $$32 \times 32 \times 3$$. The encoder is a combination of ConvNet and ResNet without batch normalization (experiment found that normalization and changing activation function perform worse since it will cause loss of information). It has 128 hidden channels in the ConvNet and 2 resnet block layers. Inversely, the decoder is a combination of transposed convolution layer and ResNet, with 128 hidden channels and 2 residual block layers. Before entering the vector quantization module, an additional ConvNet will downsample the features. The vector quantization module has 512 embedding vectors, each has 64 elements. The discriminator is a normal CNN with batch normalization and fully-connected layers. Each ConvNet layer is activated by a leaky relu function. During training, we apply a 6 epoch warm up to ignore adversarial loss and obtain an improvement of performance by decreasing loss from 0.15 to 0.12. 

When training the transformer, we encode the prediction output from the transformer into one hot matrix and calculate the cross entropy loss of each step. Since we only have 64 elements in the input sequence, the transformer will predict the next element based on the previous 1~63 elements. Fig 5 shows the performance, where the first image is the generate output, the second one is the original decoded image, and the third one is the cropped input image. There is some degree of prediction, even though it is not that well.

![Perform]({{ '/assets/images/team02/transformer_perform.png' | relative_url }})
{: style="width: 100%; max-width: 100%;"}
*Fig 5. Image recovery based on 1/8 cropped image*.

### PaintTransformer

## Demo

Code links [here](https://drive.google.com/drive/folders/1UaRDP9XtW14AJFQre5-XwW14m9Ia8YNs?usp=sharing)

## Reference
[1] Xu, Tao, et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Liu, Songhua, et al. "Paint transformer: Feed forward neural painting with stroke prediction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[3] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[4] Kim, Gwanghyun, and Jong Chul Ye. "Diffusionclip: Text-guided image manipulation using diffusion models." arXiv preprint arXiv:2110.02711 (2021).

[5] Michela Massi - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=80177333

[6] https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

[7] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

[8] https://d2l.ai/chapter_attention-mechanisms/transformer.html

[9] https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/#perception

[10] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

## Code Repository
[1] [VQGAN + CLIP Art generation](https://github.com/nerdyrodent/VQGAN-CLIP)

[2] [AttnGAN](https://github.com/taoxugit/AttnGAN)

[3] [PaintTransformer](https://github.com/wzmsltw/PaintTransformer)

---
