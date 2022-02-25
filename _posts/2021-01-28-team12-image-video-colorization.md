---
layout: post
comments: true
title: Video Colorization
author: Tony Xia, Vince Ai
date: 2022-01-28
---


> Historical videos like old movies are all black and white before the invention of colored cameras. However, have you wondered how the good old time looked like with colors? We will attempt to colorize old videos with the power of deep generative models.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## 1. Introduction
Colorization is the process of estimating RGB colors from grayscale images or video frames to improve their aesthetic and perceptual quality. Image colorization has been one of the hot-pursuit problems in computer vision research in the past decades. Various models has be proposed that can colorize image in increasing accuracy. Video colorization, on the other hand, remains relatively unexplored, but has been gaining increasingly popularity recently. It inherits all the challenges faced by image colorization, while adding more complexities to the colorization process. 

In this project, we will cover some of the state-of-the-art models in video colorization, examine their model architecture, explain their methodology of tackling the problems, and compare their effectiveness. We will also identify the limitations of current models, and shed light on the future course of research in this field. 

## 2. Problem Formulation
Formally, video colorization is the problem where given a sequence of grayscale image frame as input, the model aims to recover the corresponding sequence of RGB image frame. To simplify the problem, we usually adopts YUV channels, where only 2 channels needs to be predicted instead of 3. Video colorization poses the following challenges to the research community:

1. Like image colorization, it is a severely ill-posed problem, as two of the 3 channels are missing. They have to be inferred from other sementics of the image like object detection.

2. Unlike image colorization that only needs to take care of one image at a time, video colorization needs to remain temporally consistent when colorizatoin a sequence of video frames. Directly applying image colorization methods to videos will cause flickering effects. 

3. While image colorization is stationary, video colorization has to deal with dynamics scenes, so some frames will be blurred and hard to colorize.

4. Video colorization requires much more computing power than image colorization as the dataset are usually huge and the models are more complex.

## 3. Image Colorization


### 3.1 Instance Aware Image Colorization (2020)
InstColor proposed novel network architecture that leverages off-the-shelf models to detect the object and learn from large- scale data to extract image features at the instance and full-image level, and to optimize the feature fusion to obtain the smooth colorization results. The key insight is that a clear figure-ground separation can dramatically improve colorization performance.

The model consists of three parts:

1. an off-the-shelf pretrained model to detect object instances and produce cropped object images

2. two backbone networks trained end-to-end for instance, and full-image colorization 

3. a fusion module to selectively blend features extracted from different layers of the two colorization networks

![InstColor]({{ '/assets/images/team12/instColorModel.png' | relative_url }})

## 4. Temporal Consistency
Coloring each pixel individually using an image colorization technique often results in color flickering, as the frames do not have a way to "sync up" across time. Thus, we need a method to synchronize the pixel values across the time dimension.

### 4.1 Blind Video Temporal Consistency via Deep Video Prior (2020)

In 2020, Lei and Xing [4] proposed a novel method to address temporal inconsistency. Unlike Lai's approach [6], their network does not compute optical flow to establish correspondence. They claimed that the correspondences between frames can be captured implicitly by the visual prior coming from the CNN architecture. As corresponding patches should have similar appearances temporally, they should result in similar CNN outputs.  
Moreover, their approach does not require training on massive datasets and can be trained directly on the test video.

The authors approached the problem of temporal inconsistency by first classifying the inconsistency problem into two types:  
1. For each pixel location, the pixel values fluctuate around one single value over the entire duration of the video.  (unimodal)  
2. For each pixel location, the pixel values fluctuate around more than one single value over the entire duration of the video. (multimodal)  

To attack the first problem, the authors designed the network shown in Figure 1  
![Model-architecture]({{ '/assets/images/team12/Consistency_2.png' | relative_url }}){: style="height: 400px; max-width: 100%;"}  
*Fig 1. Lei and Xing's Method*.    
In each epoch, the model takes in each original black and white frame of the video and passes the frame through both an image colorization algorithm $$f$$ and a convolutional autoencoder $$\hat{g}$$. The authors used a U-Net[5] for this particular task, and the architectual details are shown in Figure 2. For the purpose of this study, we will explore some other CNN architectures and their efficacy later.  
![Model-architecture]({{ '/assets/images/team12/U-net.drawio.png' | relative_url }}){: style="height: 400px; max-width: 100%;"}
*Fig 2. Network design*. 

For the frame $$I_t$$ at time step $$t$$, we compute $$P_t = f(I_t), O_t = \hat{g}$$ and calculate the L1 distance between them. Then, we apply gradient descent to minimize the L1 distance. As we go through the video, the neural net $$\hat{g}$$ will approach $$f$$. As long as we stop before the neural net overfits, the unimodal inconsistency can be reduced very nicely.  

This approach, however, does not solve multimodal inconsistency. As the pixel values cluster around two or more centers, the trained network $$\hat{g}$$ would output a value in between all centers (modes). If the modes are far apart, the resulting output color would be far from any of the modes, resulting in a wrong coloring of the frame. To solve this issue, the authors proposed Iteratively Reweighted Training (IRT). Instead of producing one output image from the network, we can produce two output frames, one representing the main mode we want ($$O^{main}_t$$), while the other one represents the other modes that we want to eliminate (outlier frame, $$O^{rest}_t$$). We compute a confidence mask $$C_t$$, determining whether a specific pixel on the main frame resembles $$P_t$$ more closely than the corresponding pixel on the outlier frame.  

$$
C_t =  \begin{cases} 
1, & \text{main frame is closer}\\
0, & \text{otherwise}
\end{cases}
$$  

Define loss function $$L$$ as the follows:

$$
L = L_1(O^{main}_t \odot C_t, P_t \odot C_t) + L_1(O^{rest}_t \odot (1-C_t), P_t \odot (1-C_t))
$$

The added degree of freedom allows us to optimize $$O^{main}$$ while not compromising correctness to fit multiple modes. In order for the network to actually converge to one of the modes, in practice, we need to go through multiple iterations on the first frame, allowing the network to "grow to like" the mode, and proceed with the rest of the frames.

The Training loop is included below for reference.

```python
for epoch in range(1,maxepoch):
    for id in range(num_of_frames): 
        if with_IRT:      
            # initialize IRT
            if epoch < 6 and ARGS.IRT_initialization:
                # get the I_t and P_t
                net_in,net_gt = frames[0]
                prediction = net(net_in)
                
                # give the main frame a higher weight
                loss = L1(prediction[:,:3,:,:], net_gt) 
                            + 0.9*L1(prediction[:,3:,:,:], net_gt)

            # we've already initialized IRT, proceed with the other frames
            else:
                net_in,net_gt = frames[id]
                prediction = net(net_in)
                
                main_frame = prediction[:,:3,:,:]
                rest_frame = prediction[:,3:,:,:]
                diff_map_main,_ = torch.max(torch.abs(main_frame - net_gt) 
                                    / (net_in+1e-1), dim=1, keepdim=True)
                diff_map_rest,_ = torch.max(torch.abs(rest_frame - net_gt) 
                                    / (net_in+1e-1), dim=1, keepdim=True)
                confidence_map = torch.lt(diff_map_main, 
                                        diff_map_rest)
                                        .repeat(1,3,1,1)
                                        .float()
                crt_loss = L1(main_frame*confidence_map, 
                                net_gt*confidence_map) 
                            + L1(rest_frame*(1-confidence_map), 
                                    net_gt*(1-confidence_map))
        else:
            net_in,net_gt = frames[id] 
            prediction = net(net_in)
            loss = L1(prediction, net_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```




## 5. Conclusion
We will complete this section once we have finished running our experiments.




## 6. Reference
Please make sure to cite properly in your work, for example:

[1] Liu, Yihao, et al. "Temporally Consistent Video Colorization with Deep Feature Propagation and Self-regularization Learning." arXiv preprint arXiv:2110.04562 (2021).  
[2] Anwar, Saeed, et al. "Image colorization: A survey and dataset." arXiv preprint arXiv:2008.10774 (2020).  
[3] Zhang, Bo, et al. "Deep exemplar-based video colorization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.  
[4] Lei, Chenyang, Yazhou Xing, and Qifeng Chen. "Blind video temporal consistency via deep video prior." Advances in Neural Information Processing Systems 33 (2020): 1083-1093.
[5] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.  
[6] Lai, Wei-Sheng, et al. "Learning blind video temporal consistency." Proceedings of the European conference on computer vision (ECCV). 2018.

---
