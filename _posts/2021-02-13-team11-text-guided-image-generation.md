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

### Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/6FFo76Nzsd8?start=4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Text-guided image generation is a natural fusion of computer vision and natural language processing. Advancements in text-guided image generation serve as important benchmarks in the development of both fields. Text-guided image generation seeks to create photorealistic images from a natural language text prompt. Such a tool would allow further creation of rich and diverse visual content at an unprecidented rate. Recently, diffusion models have shown great promise towards the creation of photorealistic images. Our project will be a detailed overview of [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741.pdf). 

GLIDE stands for 'Guided Language to Image Diffusion for Generation and Editing.' The article examines both a classifier guidance diffusion model using CLIP guidance and a classifier-free guidance diffusion model. It finds that the classifier-free guidance outperforms the CLIP guided model.The classifier-free guidance model they trained was found to be favored over the previous best text-guided image generation model [DALL-E](https://arxiv.org/abs/2102.12092) 87% of the time when evaluated for photorealism, and 69% of the time when evaluated for caption similarity. The GLIDE model supports both zero-shot generation along with text-guided editing capabilites that allow for image inpainting. In this blog article, we will focus on zero-shot image generation: text-guided image generation from a diffusion model without editing. 


![GLIDE]({{ '/assets/images/team11/Examples.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Example outputs of fully-trained GLIDE with image inpaining* [1].

## Background Architecture
The GLIDE model begins by encoding the text prompt. It first encodes the input text into a sequence of K tokens which are fed into a Transformer model to generate token embeddings. The final token embedding is then fed into an augmented ADM model in place of class embeddings, and the last layer of token embeddings (K feature embeddings) is separately projected to the dimensionality of each attention layer throughout the ADM model, and then concatenated to the attention context at each layer. ADM stands for ablated diffusion model.

The original text prompt is also fed into a smaller transformer model which generates a new set of token embeddings. These embeddings and the 64x64 output of the adapted ADM model are then fed into an upsampling diffusion model with similar residual embedding connections, which will output a 256x256 model generated image.

### Transformer Model for Text to Token Encoder

![Transformer]({{ 'assets/images/team11/transformerModel.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig 1. Example of transformer model*.

Before inputting out natural language prompt into our transformer model, we must first tokenize it. HuggingFace is by far the market leader for tokenizer applications. Tokenization is the process of encoding a string of text into transformer-readable token ID integers. These token ID integers can be seen as indexes into our total vocabulary list. Once we have this tokenized text input, we then input this into a transformer model in order to generate token embeddings for each input token. These embeddings are a numerical representation of the meaning of the word given both the context and token id. As words can have multiple different meanings in different contexts and may refer to concepts in previous or future parts of the input text, we must develop a sophisticated model to ensure our token embeddings are accurate.

Transformer Models with an attention mechanism through and encoder and decoder rose to fame for this application following the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). Transformers fall under the category of sequence-to-sequence models, in which one sequence, our tokenized input, is translated into a different sequence. We can train our model for token embeddings by masking some inputs and predicting the masked words in the decoder block or through next sentence prediction. The transformer model can be broken into two stages, an encoder and a decoder. After training, we can use the encoder block to generate out token embeddings simply by taking the output of the encoder stack. 

![Encoder]({{ 'assets/images/team11/encoder1.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig 1. Example of a single encoder module*.

The encoder is a stack of encoder modules that sequentially feed into each other to output a final attention and contect vector for the decoder. The orginal tokenized input is first masked, by removing certain words from the input sentence. This masked embedding is put into a self-attention layer which is then normalized and fed through a feed forward neural network with residual connections. Each self-attention layer has three parameters, a query matrix Q, a key matrix K, and a value matrix V. GLIDE uses multi-head attention where each original tokenized input is copied and changed slighlty to include different context or positonal information and fed into our encoder. This can be seens as multiple Q, K, and V vectors that generate seperate output matrices that are concatenated together before being fed into the feed forward layer. This allows the model to learn more nuanced meanings(embeddings) for each word.

![MultiHead]({{ 'assets/images/team11/multihead_attention.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig 1. Example of multi-head attention*.

Each word will query every other word based on their keys to decide its attention on the words. The query and key vectors for each tokenized input word will then be multiplied and normalized by the square of its size and softmaxed to generate a probability distribution. This is then seen as a score per word of the same length of the input question. This score can be seen has how much that word should pay attention to every other word to inform its embedding meaning. We then multiply this score to the value vector for that word to focus on the words we want to attend to and ignore the others. Our previous input is then added back to this output to emphasize the word to focus more on itself through a residual skip connection.

The initial input into our encoder is our masked tokenized input along with a positional encoding. This positional encoding can be modeled in different ways, and is added to our initial tokenized input to create our transformer input.

After the encoder is finished, a final Key(K) and value(V) matrix is generated and sent to each encode-decode block in the decoder. After training we can throw away the decoder block and keep the endoer block. We then put our use captions that were tokenized into the encoder stack and take the value matrix (NxV) as our token embeddings. The self-attention for the decoder blocks is similar to the encoder but attempts to predict our masked caption. The output of each decoder itteration is fed shifted as input back into the decoder, so it is able to pay attention to all previous outputs when estimating the next word. It has a self-attention layer identical to our encoder blocks, followed by an encoder attention layer, in which it uses a similar process on the K and V matrices from the encoder block to pay attention to the previous caption input.

The Encoder-Decoder Attention is therefore getting a representation of both the target sequence (from the Decoder Self-Attention) and a representation of the input sequence from our encoder final K and V matrices. It produces a representation with the attention scores for each target sequence word that captures the influence of the attention scores from the input sequence. As this passes through all the Decoders in the stack, each Self-Attention and each Encoder-Decoder Attention also add their own attention scores into each word’s representation. We then compare the predicted ouputs for our masked tokenized words to the true ones and reupdate the model. A similar training stage can be done with next sentence prediction rather than a masked tokenized input.

After this is trained, we use the encoder block on out tokenized caption input to extract token embeddings that will be fed into our downsampling then upsampling diffusion models.

For the text encoding Transformer, GLIDE uses 24 residual blocks of width 2048, resulting in roughly 1.2 billion parameters.





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

![DiffusionStep]({{ '/assets/images/team11/diffusion_step.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Example Step of Diffusion Model* 

To train this, we can generate samples $$ x_t$$ approximated by $$ q(x_t | x_0)$$ by applying guassian noise to $$x_0$$ then train a model 
$$\epsilon_{\theta}$$ to predict the added noise using a surrogate objective. In a basic diffusion model, a simple standard mean-squared error loss can be used. So the outputs and inputs of our convolutional neural network will be seen as inputs and outputs of our diffusion model.

![DiffusionModel]({{ 'assets/images/team11/diffusionModel.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Example of Diffusion Model* 

GLIDE uses a more efficient version of this where $$ \sum_{\theta} $$ and $$ \mu_{\theta}$$ are learned and fixed allowing for much less diffusion steps and a faster training time.



### Classifier Free Guidance Loss Functions

Classifier free guidance guides a diffusion model without requiring a seperate classifier model to be trained. Classifier-free guidance allows a model to use its own knowledge for guidance rather than the knowledge of a classification model like [CLIP](https://github.com/openai/CLIP), which generates the most relevant text snippet given an image for label assignment. 

In the paper, a CLIP guided diffusion model is compared to a classifier-free diffusion model and finds the classifier-free diffusion model to return mor ephotorealistic results.

For classifier-free guidance with generic text prompts, we replace text captions with an empty sequence at a fixed probability and then guide our prediction towards the true caption (c) using a modified preditiction $$ \tilde{\epsilon}$$

$$  \tilde{\epsilon_{\theta}}(x_t|c) = \epsilon_{\theta}(x_t| \emptyset) + s \cdot (\epsilon_{\theta}(x_t|c) - \epsilon_{\theta}(x_t|\emptyset)) $$




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

In the training process, we first trained a text-conditional model. Then the base model is fine-tuned for classifier-free guidance.
Model Architecture:

1. ADM 64x64

- model width: 512 channels
- 2.3 billion parameters
- Transformer Model
  - 24 residual blocks of width 2048
  - 1.2 billions
- Iterations: 2.5M
- Batch Size: 2048

2. Upsampling diffusion model:

- model width increases to 384 channels
- text encoder with width 1024
- Transformer Model
  - 24 residual blocks of width 1024
- Iterations: 1.6M
- Batch Size: 512

### Dataset

The GLIDE paper simply uses the same dataset used by DALL-E. According to the openai DALL-E [github](https://github.com/openai/DALL-E/blob/master/model_card.md), "The model was trained on publicly available text-image pairs collected from the internet. This data consists partly of Conceptual Captions and a filtered subset of YFCC100M." [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) is a google dataset with 3.3 million images and annotation. [YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/about) is a list of photos and videos. The exact dataset used by both DALL-E and GLIDE has not been released.

### Training Process

1. Base Model
   First step is to encode text into a sequence of K tokens. Then, these tokens are fed into the Transformer model.

2. Upsampling Diffusion model:
   The upsampling model has the same training process as the base model with some changes in architecture parameters.

3. Classifier-free Guidance Model
   The training process of the classifier-free guidance model is the same as the base model, except that 20% of the text token sequences are replaced to empty sequence.

## Evaluation
In the evaluation process, quantitative metrics, such as Precision/Recall, IS/FID and CLIP score were used. Here we present some simple definitions of each metric:
1. Precision: fraction of relevant images among all generated images.
2. Recall: fraction of relevant images that were generated.
3. Inception Score (IS)
The inception score measures 2 things in essence:   

1) **Image quality**: Does the image clearly belong to a certain category?   

2) **Image diversity**: Do the images generated have a large variety?   

To evaluate the IS score, we first pass the generated images into a classfier, and we will get a probability distribution of the image belonging to each category. 
   
To test on _image quality_, the output probability distribution (or conditional probability distribution) of the image should have a low entropy.

To test on _image diversity_, integrating all the probability distribution of generated images (or marginal probability distribution ) should have a high entropy.

Lastly, we combine the conditional probability and marginal probability using Kullback-Leibler divergence, or KL divergence. The KL divergence is then summed over all images and averaged over all classes.

4. Frechet Inception Distance (FID)
Built upon IS, FID aims to measure the photorealism of generated images. It also requires the use of a classifier and feeding the generated images and real images into the classifier. Instead of getting the actual probability distribution, we get rid of the last output layer and use the features of the model, or outputs of the last activation layer. Then we compare the characteristics of the features of generated images and that of the real images. The acutal FID score is the calculated distance between the two feature vectors.
A lower FID score means better generated image phtorealism.

5. CLIP
CLIP score is defined as $$ E[s(f(image)·g(caption))] $$
were expectation is taken over the batch of samples, s is the CLIP logit scale, f(image).

Human evaluators were also employed to judge on the photorealistism and caption similarity of the generated images.

### Results
From human evaluations, we can see that the classifier-free guidance model outperforms the CLIP-guided model.
![scores_V]({{ 'assets/images/team11/scores_vertical.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Scores from human evaluations* [1].

From quantative metrics, we can see that there is a trade-off between the three pairs of metrics as we increase the guidance scale.

![scoresH]({{ 'assets/images/team11/scores_horizontal.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Comparing the diversity-fiedelity trande-off of classifier-free guidance and CLIP guidance* [1].


## Demo
OpenAI has released a smaller public glide model that filtered out people, faces, and nsfw content. 

![GLIDECOMPARISON]({{ '/assets/images/team11/full_vs_filtered_glide.JPG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Comparison of full GLIDE model vs Filtered* [1].

If you want a quick demo without having to code, github user valhalla has graciously created an interactive website you can try.
 [Interactive Website Link(no coding required, but slower runtime)](https://huggingface.co/spaces/valhalla/glide-text2im) 

OpenAi also released a colab file to play around with their filtered smaller model [here](https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb).


Here is a google colab [file](https://colab.research.google.com/drive/1hB2CznykUtMb5MYcEOoTjGgqKC0jPc8o?usp=sharing) we assembled using the OpenAi file and a community recreated version for DALL-E. OpenAi has not released their DALL-E model either for ethical reasons, but the community has tried to reproduce smaller versions of it.  The purpose of putting both in our colab file is to easily compare the outputs of filtered GLIDE vs recreated DALL-E. Both GLIDE and DALL-E are the current state of the art zero-shot image generation models, known for their photorealism and diverisity in image generation.



<!--
Using this released colab file from openai and a 

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

[7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv:1706.03762, 2017.

---
