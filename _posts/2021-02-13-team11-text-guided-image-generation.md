---
layout: post
comments: true
title: Text Guided Image Generation
author: Devin Yerasi, Jing Zou
date: 2022-01-18
---


> Text-guided image generation is an important milestone for both natural language procesing and computer vision. It seeks to use natural language prompts to generate new images or edit previous images. Recently diffusion models have been shown to produce better results than GANS in regards to text-guided image generation. In this article, we will be examining GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## Introduction

Text-guided image generation is a natural fusion of computer vision and natural language processing. Advancements in text-guided image generation serve as important benchmarks in the development of both fields. Text-guided image generation seeks to create photorealistic images from a natural language text prompt. Such a tool would allow further creation of rich and diverse visual content at an unprecidented rate. Recently, diffusion models have shown great promise towards the creation of photorealistic images. Our project will be a detailed overview of [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741.pdf). 

GLIDE stands for 'Guided Language to Image Diffusion for Generation and Editing.' The article examines both a classifier guidance diffusion model using CLIP guidance and a classifier-free guidance diffusion model. It finds that the classifier-free guidance outperforms the CLIP guided model.The classifier-free guidance model they trained was found to be favored over the previous best text-guided image generation model [DALL-E](https://arxiv.org/abs/2102.12092) 87% of the time when evaluated for photorealism, and 69% of the time when evaluated for caption similarity. The GLIDE model supports both zero-shot generation along with text-guided editing capabilites that allow for image inpainting. In this blog article, we will focus on zero-shot image generation: text-guided image generation from a diffusion model without editing. 


![GLIDE]({{ '/assets/images/team11/Examples.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Example outputs of fully-trained GLIDE with image inpaining* [1].

## Background Architecture
The GLIDE model begins by encoding the text prompt. It first encodes the input text into a sequence of K tokens which are fed into a Transformer model to generate token embeddings. The final token embedding is then fed into an augmented ADM model in place of class embeddings, and the last layer of token embeddings (K feature embeddings) is separately projected to the dimensionality of each attention layer throughout the ADM model, and then concatenated to the attention context at each layer. ADM stands for ablated diffusion model.

The original text prompt is also fed into a smaller transformer model which generates a new set of token embeddings. These embeddings and the 64x64 output of the adapted ADM model are then fed into an upsampling diffusuin model with similar residual embedding connections, which will output a 256x256 model generated image.

### Transformer Model for Text to Token Encoder
 For the text encoding Transformer, GLIDE uses 24 residual blocks of width 2048, resulting in roughly 1.2 billion parameters.

TALK ABOUT TRANSORMER MODEL AND INCLUDE AN IMAGE OF 1

![Transformer]({{ 'assets/images/team11/transformerModel.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig 1. Example of transformer model*.

### Diffusion Model

Diffusion models are a class of likelihood-based models that sample from a Gaussian distribution by reversing a gradual noising process that can be formulated as a Markovian chain. It begins with $$x_T$$ and learns to gradually produce less-noisy samples $$x_{T-1},...,x_{1}$$ until obtaining $$x_0$$. Each reversing of x corresponds to a certain noise level, such that $$x_t$$ corresponds to signal $$x_0$$ mixed with some noise $$ \epsilon $$ at a ratio predetermined by t. We assume $$\epsilon$$ is drawn from a diagonal Guassian distribution to simplify our equations[5]. 


So each step of the noising process can be modeled by: 
$$ 
q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{α_t}x_{t-1}, (1 -α_t)I ) 
$$

As the magnitude of noise is small at each step but the total noise throughout the Markovian chain is large, $$x_T$$ can be approximated by a $$ \mathcal{N}(0, I)$$. 

So each step of the denoising process can be learned as:

$$
p_{\theta}(x_{t-1}|x_t) := \mathcal{N}(\mu_{\theta}(x_t), \sum_{\theta}(x_t))
$$

To train this, we can generate samples $$x_t \~ q(x_t | x_0)$$ by applying guassian noise to $$x_0$$ then train a model 
$$\epsilon_{\theta}$$ to predict the added noise using a surrogate objective. In a basic diffusion model, a simple standard mean-squared error loss can be used. 

GLIDE uses a more effecient version of this where $$ \sum_{\theta} $$ and $$ \mu_{\theta}$$ are learned and fixed allowing for much less diffusion steps.

![DiffusionStep]({{ '/assets/images/team11/diffusion_step.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Example step of diffusion model* 

### Classifier Free Guidance Loss Functions

WRITE ABOUT PAGE 5 IN GLIDE PAPER


### Main Diffusion Model (ADM model architecture with additional text token residual connections) 
The ADM architecture builds off of the U-Net CNN architecture[2]. The U-Net model uses a stack of residual layers and downsampling convolutions, followed by a stack of residual layers with upsampling colvolutions, with skip connections connecting the layers with the same spatial size. In addition, they use a global attention layer at the 16×16 resolution with a single head, and add a projection of the timestep embedding into each residual block. 

![UNet]({{ '/assets/images/team11/UNet.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. UNet Architecture* .

ADM uses this model but creates a new layer called adaptive group normalization (AdaGN), which
incorporates the timestep and class embedding into each residual block after a group normalization
operation. The layer is defined as $$AdaGN(h, y) = y_sGroupNorm(h)+y_b$$, where h is the intermediate activations of the residual block following the first convolution, and $$y = [y_s,y_b]$$ comes from a linear projection of the timestep and class embedding.

ADM also incorporates variable width with 2 residual blocks per resolution, multiple heads with 64 channels per head, attention at 32, 16 and 8 resolutions, and BigGAN residual blocks for up and downsampling.

GLIDE adapts this ADM model to use text conditioning information. So, for each noised image $$x_t$$ and text caption c, it predicts 
$$ 
p(x_{t-1}|x_t,c)  
$$

Additionally, the model width is scaled to 512 channels so it has around 2.3 billion paramters just for the visual part of the model.

### Additional Upsampling Diffusion Model

In addition to the augmented ADM model, an additional upsampling diffusion model is trained and increases image size form 64x64 to 256x256. The number of visual base channels used is 384, and a smaller text encoder with 1024 instead of 2048 width is used.

### Putting it All Together

So the natural language prompt is first tokenized and encoded. Then the image batch and encodings are fed into the text-adapted ADM model and its low resolution outputs are then fed into the upsampling diffusion model along with a new encoding of the original text input. This will output a 256x256 model generated image.

## Training

### Dataset

### Training Process

## Results

## Demo

OpenAI has released a smaller public glide model that filtered out people, faces, and nsfw content. Below are code examples of how to download, initalize, and test their smaller released model. 

![GLIDECOMPARISON]({{ '/assets/images/team11/full_vs_filtered_glide.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Comparison of full GLIDE model vs Filtered* [1].

Feel free to follow the code blocks below to play with the released smaller model.
If you want a quick demo without having to code, github user valhalla has graciously created an interactive website you can try.
 [Interactive Website Link(no coding required, but slower runtime)](https://huggingface.co/spaces/valhalla/glide-text2im) 

### Download their codebase from github

```
!git clone https://github.com/openai/glide-text2im.git

%cd /content/glide-text2im/
!pip install -e .
```

### Import required libraries

```
from PIL import Image
from IPython.display import display
import torch as torch

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
```

### Run on GPU if Possible
Please note that GPU usage is heavily encouraged, as it may take more than 20 minutes to generate an example based on your text prompt on the CPU versus around 1 minute on the GPU.

```


import multiprocessing
import torch
import os
from google.colab import output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device.type)

```

### Initialize Base Diffusion Model

```
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))
```

### Initialize Upsampling Diffusion Model

```
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

```
### Helper Function

```
def show_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))

```

### Parameters and Your Text Prompt
```
prompt = "your text prompt goes here" 
batch_size =  1 #change this depending on how many images you wish to output
guidance_scale = 3.0 

#Tune this parameter to control the sharpness of 256x256 images.
upsample_temp = 0.997 
```


### Run Model and Display Output Image

```
##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Create the classifier-free guidance tokens (empty)
full_batch_size = batch_size * 2
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

# Pack the tokens together into model kwargs.
model_kwargs = dict(
    tokens=torch.tensor(
        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
    ),
    mask=torch.tensor(
        [mask] * batch_size + [uncond_mask] * batch_size,
        dtype=torch.bool,
        device=device,
    ),
)

# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

# Sample from the base model.
model.del_cache()
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model.del_cache()

# Show the output
show_images(samples)

##############################
# Upsample the 64x64 samples #
##############################

tokens = model_up.tokenizer.encode(prompt)
tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
    tokens, options_up['text_ctx']
)

# Create the model conditioning dict.
model_kwargs = dict(
    # Low-res image to upsample.
    low_res=((samples+1)*127.5).round()/127.5 - 1,

    # Text tokens
    tokens=torch.tensor(
        [tokens] * batch_size, device=device
    ),
    mask=torch.tensor(
        [mask] * batch_size,
        dtype=torch.bool,
        device=device,
    ),
)

# Sample from the base model.
model_up.del_cache()
up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
up_samples = diffusion_up.ddim_sample_loop(
    model_up,
    up_shape,
    noise=torch.randn(up_shape, device=device) * upsample_temp,
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model_up.del_cache()

# Show the output
show_images(up_samples)
```
<!--Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)-->

<!--
## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
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
-->
## Reference

[1] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I. and Chen, M., 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741.

[2] Dhariwal, P. and Nichol, A. Diffusion models beat gans on
image synthesis. arXiv:2105.05233, 2021.

[3] Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. arXiv:2006.11239, 2020.

[4] Ho, J. and Salimans, T. Classifier-free diffusion guidance.
In NeurIPS 2021 Workshop on Deep Generative Models
and Downstream Applications, 2021. URL https://
openreview.net/forum?id=qw8AKxfYbI.

[5] Nichol, A. and Dhariwal, P. Improved denoising diffusion
probabilistic models. arXiv:2102.09672, 2021.

[6] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. arXiv:2102.12092, 2021.

---
