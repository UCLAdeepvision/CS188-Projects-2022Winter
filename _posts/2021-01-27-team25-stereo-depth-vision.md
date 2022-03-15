---
layout: postmathjax
comments: true
title: Depth From Stereo Vision
author: Alex Mikhalev, David Morley
date: 2022-01-27
---


> Our project explores the use of Deep Learning for inferring the depth data based on two side-by-side camera images. This is done by determining which pixels on each image corresponding to the same object (a process known as stereo matching), and then calculating the distance between corresponding pixels, from which the depth can be calculated (with information about the placement of the cameras). While there exist classical vision based solutions to stereo matching, deep learning can produce better results.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Background
Stereo matching is the process of aligning two images taken by distinct cameras of the same object. In the very simple case,
with perfect images, and one object at constant depth, one can compute the disparity (pixel alignment offset) required to align both the left and right camera images. This information can be used, along with the distance between the two cameras, to compute a distance estimation for this object. In the real world case, this problem becomes much more complicated. Many objects reside in the scene with different textures, shadows, etc, and performing an alignment between all these points is difficult, making it more challenging to estimate distance than in the simple case. There are two different forms of stereo matching: active and passive. In the active case, one simplifies the problems of alignment by projecting light (ofter laser dot matrix) and using other adaptive mechanisms to make it easier to align the two camera images. This hardware is much more expensive however and thus not as likely to see widespread use. Passive stereo imaging just involves two statically placed cameras and thus is a much harder problem that we will explore for our final project.

A natural question that also might be asked, is why not use a single camera for depth estimation? There are some models that explore this, but it is much more difficult to get accurate depth measurements, and there is a large gap between depth accuracies (as shown in 4)).

## Paper Choice
We evaluated multiple different papers when deciding on what we wanted to choose for our final project. As we purchased an OAK-D Lite stereo camera for use in our actual environment, we favored algorithms that could run in close to real time, so we'd be able to observe the depth estimations of objects as we walked
by them. We first considered the Pyramid Matching Network, which had a rather simple multilayer convolutional model, but weren't the most confident in its real time performance, given the author's reliance on a fairly powerful GPU. We then considered both HITNet (a recent network, although admittedly a somewhat complex one) authored this year Google, and StereoNet (a model which uses a 2 pass Siamese Network). We decided to do a deep dive into HITNet due to its more complex design, and compared our results to the pretrained HITNet model, along with an implementation of StereoNet. Below we give an explanation of the HITNet Model.

## HITNet
HITNet is a recent model that works to use some of the recent techniques from active stereo depth applied to the passive stereo problem. It is  optimized for speed, using a combination of a fast multi-resolution initial step, and 2d disparity propagation, instead of much more costly 3D convolutions. Although it performs slightly worse than the 3D convolutional models, it takes only milliseconds to process an image pair, compared to seconds from those models.

![UNET]({{ '/assets/images/team25/unet.png' | relative_url }})

This is the UNet feature extractor implementation used in the HITNet model.
It has 4 stages of downsampling and upsampling.
It also uses LeakyReLU activation with 0.2 slope for improved training.

```python
class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input, sc):
        x = self.up_conv(input)
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(3, C[0], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_1 = nn.Sequential(
            SameConv2d(C[0], C[1], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[1], C[1], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_2 = nn.Sequential(
            SameConv2d(C[1], C[2], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[2], C[2], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_3 = nn.Sequential(
            SameConv2d(C[2], C[3], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[3], C[3], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_4 = nn.Sequential(
            SameConv2d(C[3], C[4], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.up_3 = UpsampleBlock(C[4], C[3])
        self.up_2 = UpsampleBlock(C[3], C[2])
        self.up_1 = UpsampleBlock(C[2], C[1])
        self.up_0 = UpsampleBlock(C[1], C[0])

    def forward(self, input):
        x0 = self.down_0(input)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        o0 = self.down_4(x3)
        o1 = self.up_3(o0, x3)
        o2 = self.up_2(o1, x2)
        o3 = self.up_1(o2, x1)
        o4 = self.up_0(o3, x0)
        return o4, o3, o2, o1, o0
```

<center> Figure 1: Example U-Net Implementation </center>

### Initialization
The key idea of the initialization step of HITNet is the use of an encoder decoder network, with the model using the results of all the different resolution layers output by the decoder. The initialization phase builds on top of U-Net (such a decoder-encoder network) which is shown in Figure 1. After generating different resolution features for both $$I_R$$ and $$I_L$$ (the left and right images) we obtain two multiscale representations denoted $$ \epsilon^R $$ and $$\epsilon^L$$. Taking $$\epsilon^R$$ and $$\epsilon^L $$ we then attempt to align tiles of these images. The idea here is for each resolution, we want to tile that image and map tiles in $$\epsilon^L$$ to $$\epsilon^R$$. We denote the feature map for a specific resolution l as $$e_l$$. To get the tile features we run a 4x4 convolution on both $$\epsilon^L$$ and $$\epsilon^R$$, but there's a subtle point here: We want to cover our features with overlapping tiles, but we also want to minimize the number of disparity values we have to try. In order to solve this problem the authors introduce the following asymmetry, when applying the convolution on the selected tile regions from the feature map. For $$\epsilon^L$$ overlap the tiles in x direction and use a 4x4 stride on the convolution. For $$\epsilon^R$$ do not overlap the tiles, but instead use a 4x1 stride in the convolution so the tiles overlap in the convolution computation. This allows us to formulate our matching cost c for a specific location (x,y) with resolution l and disparity d as:
$$c(l,x,y, d) = \lvert \lvert e_{l,x,y} - e_{l, 4x -d, y} \rvert \rvert_1$$
We then compute the disparities for each (x,y) location and resolution l, where D is the max disparity, noting that our convolution trick allows us to try far fewer values for the disparity with the following:

$$d_{l,x,y}^{init} = argmin_{d \in [0,D]}c(l,x,y,d)$$

This search is exhaustive over all potential disparity values.

The authors also add an additional parameter to the model which they denote the tile feature descriptor $$p_{l,x,y}^{init}$$ for each point (x,y) and resolution l. This is a value which the output of a perceptron and leaky ReLU fed the costs of the best matching disparity d, and the embedding for that specific feature at that point. The idea of this feature is to pass along the confidence of the match to the network laters.

![HITNet]({{ '/assets/images/team25/HITNet.png' | relative_url }})

<center> Figure 2: Details of Propagation and Initialization Steps </center>


```python
@functools.lru_cache()
@torch.no_grad()
def make_warp_coef(scale, device):
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index)
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y


def disp_up(d, dx, dy, scale, tile_expand):
    n, _, h, w = d.size()
    coef_x, coef_y = make_warp_coef(scale, d.device)

    if tile_expand:
        d = d + coef_x * dx + coef_y * dy
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(n, 1, h * scale, w * scale)
    return d
```

<center> Figure 3: Implementation of the Warp Step </center>
### Propagation

#### Warping
The second step of the model, propagation, uses input from all the different resolutions of the features. First we perform a warping between the individual tiles that are associated with each feature resolution using the computed disparity values from before. We warp the tiles in the right tile features $$e^R$$ to the left tile features $$e^L$$ (converting them into their original size on the feature map) using linear interpolation along the scan lines. To pass this information on to the next iteration of the algorithm a cost vector is also computed which takes the magnitude of the distance between all of the feature values in this 4x4 tile.

#### Update
A CNN model $$U$$ then takes n tile hypotheses as input and predicts a change for the tile plus a confidence value $$w$$. This convolutional layer is useful as it allows the use of neighboring information from other tiles, along with the multidimensional inputs to be used in updating the tile hypothesis. The tile hypothesis is augmented with the matching costs $$\phi$$ computed during the warping step, with the warp costs stored for the current estimate and offset +-1 to give a local cost volume. In the paper the augmented tile map is denoted as follows:

$$a_{l,x,y} = [h_{l,x,y}, \phi(e_l, d - 1), \phi(e_l, d), \phi(e_l, d +1)]$$

Our model then outputs deltas for each of the n tile hypothesis maps and new confidence values as shown here:

$$(\Delta h_l^1, w^1, \cdots, \Delta h_l^n, w^n) = U_l(a_l^1, \cdots, a_l^n; \theta_{U_l})$$

The CNN model $$U$$ is made or resnet blocks followed by dilated convolutions. The update is performed starting at lowest resolution feature, moving up, so at the second resolution feature there are now two inputs, and then three, and so on and so forth. This action can be seen in Figure 2. Tiles are upsampled as they move up layers to maintain they all stay the appropriate dimension. At each location the hypothesis with the largest confidence is selected until the starting resolution is reached. They then run the model one additional time on the optimal tiles for further refinement and output this as the disparity map result.

### Loss
The network uses multiple different losses: an initialization loss, propagation loss, slant loss, and confidence loss. These values are all weighted equally and used to optimize the model.

## StereoNet
The main insight of StereoNet is that we can aggressively downsample the input disparity images so we can perform a much less costly alignment between two significantly reduced feature sizes. Unlike HITNet which saves computational costs by omitting 3D convolutions, StereoNet still makes these computations, just at a severely reduced resolution. The network uses a simple decoder encode approach, this time with a Siamese network (a neural network where there are
two separate input vectors, connected to separate networks, but both of these networks have identical weights). Performing the cost volume computation on this vastly reduced representation of the input images allows the network to train/run much faster, but still attains quite good performance.


![StereoNet]({{ '/assets/images/team25/StereoNet.png' | relative_url }})
<center>Figure 4: The overall structure of StereoNet</center>

### Feature Network
The Siamese network first aggressively downsamples the two input images using a series of K 5x5 convolutions with a stride of 2 and 32 channels. Residual blocks with 3x3 convolutions, batch-normalization, and Leaky ReLus are then applied, followed by a 3x3 convolution layer, without any activation or activation. This outputs a 32-dim vector representation for each down sampled point. Through this representation there is a low dimensional representation of the input that stil has a fairly large receptive field.

### Cost Volume
After downsampling, like most conventional deep learning disparity models, the network computes a cost volume along scan lines (evaluates every possible offset in the x direction up to the value of the maximum disparity). The cost volume is then aggregated with four 3D convolutions of size 3x3x3, and once more batch-normalization + leaky ReLu activations. Finally an additional 3x3x3 layer without batch normalization + leaky ReLu is applied.

```python
def make_cost_volume(left, right, max_disp):
    cost_volume = torch.ones(
        (left.size(0), left.size(1), max_disp, left.size(2), left.size(3)),
        dtype=left.dtype,
        device=left.device,
    )

    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    return cost_volume
```
<center> Figure 5: Implementation of Cost Volume Computation </center>

### Differentiable Arg Min
Optimally the disparity with the mininum cost for each pixel would be selected but such a selection is not differentiable. The two approaches tried in the paper are a softmax combination of the selected disparity values and a probabilistic sampling during training over the softmax distribution to approximate the softmax function. This probabilistic approach performed much worse in the author's results, so we stuck with the first selection instead.

The first (and much better performing) loss function.

$$ d_i = \sum_{d=1}^D d \cdot \frac {\text{exp}(-C_i(d))} {\sum_{d'} \text{exp}(-C_i(d'))}$$

The probabilistic loss function.

<center> $$ d_i = d, \text{where}$$</center>

<center> $$d \sim \sum_{d=1}^D d \cdot \frac {\text{exp}(-C_i(d))} {\sum_{d'} \text{exp}(-C_i(d'))}$$</center>

### Upsampling
To upsample the image the disparity map is first bilinearly upsampled to the output size. It then is passed through the refinement layer which first consists 3x3 convolutiion. This is followed by 6 residual blocks with 3x3 dilated convolutions (to increase the receptive field), with dilation factors of 1,2,4,8, and 1 respectiveley. The final output is run through a 3x3 convolutional layer. The authors tried two different techniques, one running the upsample once and another cascading the upsample layer for further refinement.
### Loss Function
Loss is computed as the difference between the ground truth disparity and the predicted disparity at $k$, the given refinement level (k = 0 denoting output before any refinement) and is given by the following equation.

$$L = \sum_k p(d_i^k - \hat{d}_i)$$


## Replicating Results
To train both HITNet and StereoLab models ourselves, we heavily relied upon the work of GitHub user zjjMaiMai in their
repository TinyHITNet [6], which implements both models in PyTorch. This is an example result on a training image after 3200 steps of  training the HITNet model on the KITTI 2015 dataset.
The top image is the ground truth from a laser scanner pointcloud. The second image is the predicted depth.
The last two images are each of the camera views.

![KITTI HITNet trained for 3200 steps]({{ '/assets/images/team25/kitti-hitnet-3200step.png' | relative_url }})

This is an example result on a training image after 11,700 steps of training the StereoNet model on the KITTI 2015 dataset.
StereoNet has fewer parameters, requires less GPU memory, and trains faster than HITNet.

![KITTI StereoNet trained for 11,700 steps]({{ '/assets/images/team25/kitti-stereonet-11700step.png' | relative_url }})

This is the EPE error during training. The orange line is HITNet trained with batch size 1, and the blue line is
StereoNet trained with batch size 2. We can see that StereoNet converges during training much faster.
The steps themselves also ran faster than HITNet. There is still potential improvement in the optimization process.
We also did not train HITNet enough because we did not have time yet.

![Train Accuracy]({{ '/assets/images/team25/train-epe.png' | relative_url }})

## Experimental Contributions
When looking at the StereoNet Model we were somewhat curious what the effect of using a Siamese network had on the model, over training separate models for both left and right images. We surmised that there may be some differences in images depending on the side they were taken on and it might make sense to learn a different representation for each of these pieces of the network. As the feature refinement part of the network as well as the cost matrix are likely very similar in function, we only trained separate features for both model sizes. While this does effectively reduce by half our quantity of training data per weight in the feature generation of our network(as we are no longer training the weights per iteration on both the left and right images) we were hopeful it might result in a slight performance gain. We also experimented with the optimizer given in the paper, trying AdaBound in our new non-Siamese implementation to see if we could notice any improvement over the original paper.

```python
self.feature_extractor_left = [conv_3x3(3, 32, 2), ResBlock(32)]
self.feature_extractor_right = [conv_3x3(3, 32, 2), ResBlock(32)]
for _ in range(self.k - 1):
    self.feature_extractor_left += [conv_3x3(32, 32, 2), ResBlock(32)]
    self.feature_extractor_right += [conv_3x3(32, 32, 2), ResBlock(32)]
self.feature_extractor_left += [nn.Conv2d(32, 32, 3, 1, 1)]
self.feature_extractor_right += [nn.Conv2d(32, 32, 3, 1, 1)]

self.feature_extractor_left = nn.Sequential(*self.feature_extractor_left)
self.feature_extractor_right = nn.Sequential(*self.feature_extractor_right)
```
<center> Figure 6: Alterations to Implement Model as NonSiamese</center>

<!--TODO(morleyd): Look into how to offset loss to account for the nonexisting pixels in the morph-->

As camera alignment is often imperfect, we were curious if expanding the local convolution to include a y offset in the highest resolution layer of HITNet could result in a minor improvement in matching. For the pixels that are now misaligned we had to determine how to modify the loss computation. We decided to count the now nonexisting pixels as having an initialization value of 0, making it fairly unfavorable to use this offset, but still potentially better. While this would introduce a somewhat artificial penalty to choosing a displacement in the y direction, in the pedantic case, where the pixels have sharp boundaries and are all misaligned by some +-y, we would still notice a significant decrease in the loss. We thus decided this minor change was worth exploring in our modified implementation of HITNet, despite the minor increase in computational cost.

## Reference
Please make sure to cite properly in your work, for example:

[1] Zhuoran Shen, et al. ["Efficient Attention: Attention with Linear Complexities."](https://arxiv.org/pdf/1812.01243v9.pdf) *Winter Conference on Applications of Computer Vision*. 2021.

[2] Vladimir Tankovich, et al. ["HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching."](https://arxiv.org/abs/2007.12140) *Conference on Computer Vision and Pattern Recognition*. 2021.

[3] Jia-Ren Chang, et al. ["Pyramid Stereo Matching Network."](https://arxiv.org/abs/1803.08669) *Conference on Computer Vision and Pattern Recognition*. 2018.

[4] Nikolai Smolyanskiy, et al. ["On the Importance of Stereo for Accurate Depth Estimation: An Efficient Semi-Supervised Deep Neural Network Approach."](https://arxiv.org/abs/1803.09719) *Conference on Computer Vision and Pattern Recognition*. 2018.

[5] Olaf Ronneberger, et al. ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) *Conference on Computer Vision and Pattern Recognition*. 2015.

---