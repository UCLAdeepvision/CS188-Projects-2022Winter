---
layout: post
comments: true
title: Trajectory Prediction
author: Sudhanshu Agrawal, Jenson Choi
date: 2022-01-27
---

> Behavior prediction in dynamic, multi-agent systems is an important problem in the context of self-driving cars. In this blog, we will investigate a few different approaches to tackling this multifaceted problem and reproduce the work of [Gao, Jiyang et al.](https://arxiv.org/abs/2005.04259) by implementing VectorNet in PyTorch.

<!--more-->

{: class="table-of-content"}

- TOC
  {:toc}

## Introduction

Self-driving is one of the biggest applications of Computer Vision in industry. Naturally, being able to predict the trajectory of an autonomous vehicle is paramount to the success of self-driving. Our project will be an extension of [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259), which is a hierachical graph neural network architecture that first exploits the spatial locality of individual road components represented by vectors and then models the high-order interactions among all components. Research on trajectory prediction is not limited to the self-driving domain, however, [Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html) and [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892) are more generic examples of work related to multi-agents interaction forecasting which we will also explore in this project. In particular, if possible, we will attempt to extend the work on human trajectory prediction in crowded spaces to simulate the effect that social distancing due to COVID-19 has on the trajectories. We may be able to do this by exploring ways of adding a factor to create a form of 'repulsion' between the agents.

# Paper 1: VectorNet - Encoding HD Maps and Agent Dynamics From Vectorized Representation

## Introduction

Predicting the behavior of others on the road is difficult: often, a holistic understanding of a scene and its context is required, including the width of road lanes, four-way intersection rules, traffic lights, and signs. Recent learning-based approaches to tackling behavior prediction require building a representation to encode the map and trajectory information, often in the form of High Definition (HD) maps. These HD maps are rendered as color-coded attributes (Fig 1, left), which encode scene context information, such as traffic signs, lanes, and road boundaries, with ConvNets. These approaches have a few drawbacks. First and foremost, utilizing ConvNets to encode scene context information requires a lot of compute and time. In addition, processing maps as imagery makes it challenging to model long-range geometry, such as lanes merging ahead. To address these shortcomings, the authors of this paper propose to learn a unified representation for multi-agent dynamics and structure scene context directly from their vectorized form (Fig 1, right) using a hierarchical graph neural network architecture, VectorNet, with the ultimate goal of building a system which learns to predict the intent of vehicles, which are parameterized as trajectories.

![Artificial neural network]({{ '/assets/images/team13/HD_map.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
_Fig 1. Illustration of the rasterized rendering (left) and vectorized approach (right) to represent high-definition map and agent
trajectories. (Image source: <https://arxiv.org/pdf/2005.04259.pdf>)_

## Model Architecture

The authors attempt to encode various features of road networks via vectors. Road entities such as crosswalks, stop signs, lanes, and others, can be represented as points, polygons and curves. All these entities can further be represented as polylines. Finally, all these polylines can be represented as sets of vectors. For example, the trajectory of some agent could be represented as a set of vectors sampled along the path of motion.

In particular, each vector $$v_i$$ belonging to a polyline $$P_j$$ is a node in the graph with node features given by:
$$ v_i = [d_i^s, d_i^e, a_i, j]$$
Where:

- $$d_i^s$$ and $$d_i^e$$ are the coordinates of the starting and ending points of the vector
- $$a_i$$ corresponds to attribute features, such as object type (lane, stop sign, etc), timestamps for trajectories, or road features, such as speed limits.
- $$j$$ is the integer id of $$P_j$$, the polyline to which the vector belongs.

Thus, a polyline is a set of vectors :
$$ P = \{v_1, v_2\cdots, v_P\}$$

There are two types of components of the network. One type which looks at interactions between nodes within a polyline, forming polyline subgraphs, and one which looks at the higher order interactions between the polylines themselves, denoted as the global graph.

### Polyline Subgraphs

For a given polyline, $$P$$, a single layer of subgraph propagation is formulated as :
$$ v*i^{(l+1)} = \varphi*{rel}(g*{enc}(v_i^{(l)}, \varphi*{agg}(\{g\_{enc}(v_j^{(l)})\}))$$
Where:

- $$ v_i^{(l)}$$ is the node feature for the $$l$$-th layer of the subgraph network
  - Therefore, $$v_i^{(0)}$$ denotes the input features, $$v_i$$.
- $$g_{enc}$$ is a function that transforms the individual node features
  - It is implemented as a MLP with the weights shared among all the nodes
  - It has a single fully connected layer followed by layer normalization, followed by ReLU.
- $$\varphi_{agg}$$ aggregates the information from all the neighbouring nodes that surround the target node
  - It is implemented as a maxpool operation
- $$\varphi_{rel}$$ is the relational operator among the node and its neighbours
  - It is simply a concatenation operation

In summary, the overall function takes in all the neighbouring nodes to the target, encodes them as vectors, then performs a maxpool. It then encodes the current node and concatenates it with the result of the maxpooling operation.

Each layer of the network has different weights for $$g_{enc}$$, but they are shared among the nodes.

Finally, to obtain the features for a polyline, the nodes along the polyline are maxpooled as :
$$p = \varphi_{agg}(\{v_i^{(L_p)}\})$$

### Global Graph

Given a set of polyline node features, $$\{p_1, p_2, \cdots, p_P\}$$,
the objective is to model the interaction between these polylines. This is given by :

$$ \{p_i^{(l+1)}\} = \text{GNN}(\{p_i^{(l)}\}, A) $$

Where:

- The polyline node features at a layer are given by $$\{p_i^{(l)}\}$$
- $$\text{GNN}$$ is a single layer of the graph neural network
  - $$\text{GNN} (P) = \text{softmax} (P_QP_K^T)P_V$$ <br> Where $$P$$ is the node feature matrix and $$P_Q, P_K, P_V$$ are its linear projections.
- $$A$$ is the adjacency matrix for the set of polyline nodes.
  - It can be assumed that the polyline graph is a fully connected graph and $$A$$ can be computed using spatial distance as a heuristic.

Future trajectories can thus be decoded from the nodes corresponding to the moving agents
$$ v*i^{future} = \varphi*{traj}(p_i^{(L_t)}) $$
Where:

- $$L_t$$ is the total number of $$\text{GNN}$$ layers.
- $$\varphi_{traj}$$ is the trajectory decoder implemented as an MLP.

Therefore the values produced by the last layer are passed through the decoder for each polyline that is being considered.

### Auxiliary Task

An auxiliary objective is introduced to encourage the global interaction graph to better capture interactions among different trajectories and map polylines.

During training time, the model randomly masks out the features for a subset of polyline nodes, say, $$p_i$$, and the objective is to recover the masked out features as :
$$ \hat{p*i} = \varphi*{node}(p_i^{(L_t)}) $$
Where:

- $$\varphi_{node}$$ is the node feature decoder implemented as an MLP. This is not used during inference.
- $$p_i$$ is a node from a fully connected unordered graph
- To identify a polyline node when its features are masked out, the minimum values o the start coordinates of all the vectors that belong to it is computed. This becomes $$p_i^{id}$$
  - The input node features are thus $$p_i^{(0)} = [p_i; p_i^{id}]$$

This task is similar to that used by the BERT language model that predicts missing tokens based on bidirectional data.

### Overall Structure

The objective function is thus :
$$ L = L*{traj} + \alpha L*{node}$$
Where:

- $$L_{traj}$$ is the negative Gaussian log-likelihood for the ground-truth future trajectories
- $$L_{node}$$ is the Huber loss between the predicted node features and the ground-truth masked node features
- $$\alpha$$ is a scalar that balances the two loss terms

## Experiments

The authors tested the performance of VectorNet on one vehicle behavior prediction benchmark, the Agroverse dataset.

### Dataset

Argoverse motion forecasting is a dataset designed for vehicle havior prediction with trajectory histories. There are 333K 5-second long sequences split into 211K training, 41K validation and 80K testing sequences. The creators curated this dataset by mining interesting and diverse scenarios, such as yielding for a merging vehicle, crossing an intersection, etc. The trajectories are sampled at 10Hz, with (0, 2] seoncds are used as observation and (2, 5] seconds for trajectory prediction. Each sequence has one "interesting" agent whose trajectory is the precition target.

### Metrics

The authors adopt the Average Displacement Error (ADE) computed over the entire trajectories and the Displacement Error at t (DE@ts) metric, where $$t\in \{1.0, 2.0, 3.0\}$$ seconds. Additionally, the displacements are in meters.

### Results

The perfomance of VectorNet on the Argoverse dataset is compared with several baseline approaches and a few state-of-the-art architectures, which is summaried in table 1.

| Model                      |       DE@3s       |               ADE |
| :------------------------- | :---------------: | ----------------: |
| Constant Velocity          |       7.89        |              3.53 |
| Nearest Neighbor           |       7.88        |              3.45 |
| LSTM ED                    |       4.95        |              2.15 |
| Challenge Winter: uulm-mrm |       4.19        |              1.90 |
| Challenge Winter: Jean     |       4.17        |              1.86 |
| VectorNet                  | $$\textbf{4.01}$$ | $$\textbf{1.81}$$ |

_Table 1. Trajectory prediction performance on teh Argoverse Forecasting test set when number of sampled trajectories K=1. Results were retrieved from the Agroverse leaderboard on 03/18/2020_

The baseline approaches are the constant velocity baseline, nearest neighbor retrieval and LSTM encoder-decoder. The state-of-the-art approaches are the winners of the Agroverse Forecasting Challenge. From table 1, VectorNet is able to outperform both state-of-the-art models in terms of DE@3s and ADE when K=1.

### Visualization

Some of the predictions made by VectorNet on trajectories in the Agroverse dataset are visualized in Fig 2.

![Artificial neural network]({{ '/assets/images/team13/vectornet_visual.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
_Fig 2. (Left) Visualization of the prediction: lanes are shown in
grey, non-target agents are green, target agent’s ground truth trajectory is in pink, predicted trajectory in blue. (Right) Visualization of attention for road and agent: Brighter red color corresponds
to higher attention score. (Image source: <https://arxiv.org/pdf/2005.04259.pdf>)_

It is observed that when argents are facing multiple choices (first two examples), the attention mechanism is able to focus on the correct choices (two right-turn lanes in the second example). The third example is a lane-changing agent, the attended lanes are the current lane and target lane. In the fourth example, though the prediction is not accurate, the attention still produces a reasonable score on the correct lane.

## Takeaway

The main contribution of this paper is the method to encode HD maps and agent dynamics as vectors.

The paper proposes a novel hierarchical graph neural network with:

- Level 1: aggregates information among vectors inside a polyline
- Level 2: models the higher order relationships among polylines

The VectorNet is on-par in terms of performance with the best ResNet on the large scale in-house data set used by the researchers and is more computationally efficient.

It outperforms the best ConvNet on the publicly available Argoverse data set while having a lower computational cost.

## Demo

_<https://colab.research.google.com/drive/1ldrHriSrpxNQ9t2igfBTdctp9pMUSYQw?usp=sharing>_

## Reference

Please make sure to cite properly in your work, for example:

[1] Gao, Jiyang, et al. "Vectornet: Encoding hd maps and agent dynamics from vectorized representation." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2020.

[2] Alahi, Alexandre, et al. "Social lstm: Human trajectory prediction in crowded spaces." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

[3] Gupta, Agrim, et al. "Social gan: Socially acceptable trajectories with generative adversarial networks." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2018.

## Code Repository

[1] [Reimplement VectorNet](https://github.com/xk-huang/yet-another-vectornet)

[2] [Social LSTM Implementation in PyTorch](https://github.com/quancore/social-lstm)

[3] [Social GAN](https://github.com/agrimgupta92/sgan)

---
