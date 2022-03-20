---
layout: post
comments: true
title: Scene Generation With NeRF
author: Felix Zhang
date: 2022-02-24
---


> This blog will contain updates and technical explanations for Felix Zhang's CS188 Project. The focus of project is to explore various approaches to generating 3-D scenes with Neural Radiance Fields. We are planning on utilize NeRF to generate 3-D scenes with novel view points. We are currently exploring the possiblity of leverage the Infinite Nature model, but are experimenting with GPU requirements.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Team 23

### Group Members: Felix Zhang

### 3D Scene Generation (involving Generative Neural Radiance Fields)

There has been a recent surge in using exploring image generation in three dimensions due to the adoption of virtual, augmented, and mixed reality devices, as well as the possibility of using scenes to power downstream task datasets for agents. We explore in this project 3D Scene Generation, especially focusing on a recent interesting development of Generation with Neural Radiance Fields.

## Neural Radiance Fields (NeRF)

Neural Radiance Fields are a recent development introduced by [Mildenhall et al.](https://www.matthewtancik.com/nerf) for 3d image synthesis by optimizing a volumetric function with a multi-layer perceptron network. The achieve impressive results compared to traditional generative techniques. 

NeRF works by estimating 3-D RGB data from 5-D data of coordinates, radiance, and viewpoint. It uses a simple MLP to estimate the volumetric color data and then uses volume rendering techniques to generate 2-D images from the cooresponding 3-D points. This is then able to generalize well to novel viewpoints. The following figure demonstrates the NeRF process.


![NERF_TRACTOR]({{ '/assets/images/team23/nerf_tractor.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

We can see that NeRF is able to successfully render realistic new viewpoints.


![NERF_DRUMS]({{ '/assets/images/team23/nerf_drums.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


We specifically utilize the open source PyTorch implementation of NeRF located at [NeRF-Torch]. We have attached the NeRF model below, and use various helper functions as implemented to render the rays and reconstruction of the 2-D image from the 3-D volume.

```
# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
```

## Infinite Nature

For the 3-D Scene Generation portion of the model, we are consider the Infinite Nature paper by Liu et. al. The authors propose a model that is able to generate long-range novel view points of scenes by using a render-refine-repeat process. The render portion uses the disparty maps, then the refine portion infills the novel regions, and the repeat process is then able to generate more ad infinitum.


![INFINITE_NATURE]({{ '/assets/images/team23/infinite_nature.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

We resolved the dependencies and run the demo given.
[Given Demo](https://colab.research.google.com/github/google-research/google-research/blob/master/infinite_nature/infinite_nature_demo.ipynb#scrollTo=08MXs7cBPDwO) 


We are currently exploring enforcing NeRF onto infinite nature but running into GPU difficulties. We believe that NeRF should be able to help generate more convincing scene generations, and combined with Infinite Nature, should allow able to generate 3-D imagery on demand.

<!-- 
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1]. -->
<!-- 
## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

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

or you can write in-text formula $$y = wx + b$$. -->

## Reference

[1] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, Angjoo Kanazawa. "Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image." *Proceedings of the ICCV conference on computer vision and pattern recognition*. 2020.

[2] Michael Niemeyer, Andreas Geiger. "GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields" *Proceedings of the CVPR conference*. 2021.

[3] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" *Proceedings of ECCV conference* 2020.

[4] Patrick Esser, Robin Rombach, Björn Ommer. "Taming Transformers For High-Resolution Image Synthesis" *Proceedings of the CVPR conference*. 2021.

[5] Sheng-Yu Wang, David Bau, Jun-Yun Zhu. "Sketch Your Own GANs" 2021.

[6] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, Dahua Lin. "CityNeRF: Building NeRF at City Scale" 2021.

[7] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin. "FastNeRF: High-Fidelity Neural Rendering at 200FPS" *Proceedings of the ICCV conference* 2021.

[8] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever. "DALL-E: Zero-Shot Text-to-Image Generation" 2021.

## Code Repos and Pages

[1] [NeRF](https://github.com/bmild/nerf)

[2] [Infinite Nature](https://github.com/google-research/google-research/tree/master/infinite_nature)

[3] [Giraffe](https://github.com/autonomousvision/giraffe)

[4] [NeRF-Torch](https://github.com/yenchenlin/nerf-pytorch)
