---
layout: post
comments: true
title: Graph Convolution Networks for fusion of RGB-D images
author: Alexander Swerdlow, Puneet Nayyar
date: 2022-03-16
---


> 3D Point Cloud understanding is critical for many robotics applications with unstructured environments. Point Cloud Data can be obtained directly (e.g. LIDAR) or indirectly through depth maps (stereo cameras, depth from de-focus, etc.), however efficient merging the information gained from point clouds with 2D image features and textures is an open problem. Graph convolutional networks have the ability to exploit geometric information in the scene that is difficult for 2D based image recognition approaches to reason about and we test how these features can improve classification models.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


# Introduction
Our project seeks to implement and improve upon graph convolution approaches to 3D point cloud classification and segmentation. We plan to explore the dynamic graph approaches proposed in [1] and [7] which create graphs between each layer with connections between nearby points and uses an asymmetric edge kernel that incorporates relative and absolute vertex locations. This approach contrasts with the typical approach to learning on point clouds which operates directly on sets of points. PointNet [8] is one popular implementation that takes this approach, learning spatial features for each point and using pooling to later perform classification or segmentation. 

# Introduction To Graph Neural Networks

Graph Neural Networks, or GNNs, are as the name suggests, neural networks that make sense of graph informaton. This umbrella term covers graph-level tasks (which our project focuses on), node-level tasks (e.g. predicting the property of a node), and edge-level tasks. GNNs take some input graph $$G = (V, E)$$ and transform that graph in some sequence to produce a desired prediction. A basic GNN might look like several layers, each composed of some non-linear function $$f$$ (e.g. an MLP) that takes in the current graph $$\mathcal{G}_i$$. Depending on the GNN, the graph structure, commonly represented by an adjacency matrix, may stay the same throughout the layers, or change at each layer. The node and/or edge features are progessively transformed by this non-linear mapping and, if the task is graph-level classification, some pooling operation is typically performed to reduce the dimensionality.Depending on the GNN, the graph structure, commonly represented by an adjacency matrix, may stay the same throughout the layers, or change at each layer. For this project, we explored multiple implementations of GNNs, which are explained here. 

# DGCNN
Our first approach was a Graph Convolutional Network (GCN) that used the influential EdgeConv [1] operation for convolution. We follow in their approach in modifying our graph size at each layer, performing max pooling after each EdgeConv layer. We chose the Dynamic Graph CNN (DGCNN) approach since it keeps the maximum receptive field, but reduces the number of parameters required at each sucessive layer. We also describe the EdgeConv operation below for completeness.

### EdgeConv
In order to tackle the problem of maintaining a graph's permutation invariance while also exploiting the geometric relationship between nodes, Wang et al. proposed a novel differentiable approach called EdgeConv [1]. The method resembles that of typical convolutions in CNNs, where local neighborhood graphs are created for each point and convolution operations are applied on the edges in this neighborhood. Additionally, in contrast to the approaches employed by other graph based networks, these local neighborhoods change after every layer. More specifically, the graph is recomputed after each layer such that the nearest neighbors are now those closest in the feature space. These dynamic graph updates allow the embeddings to be updated globally, and the authors show that this distinction leads to the best results for point cloud classification and segmentation. 

Formally, at each layer $$l$$, a directed graph $$\mathcal{G^{(l)}} = (\mathcal{V^{(l)}}, \mathcal{E^{(l)}})$$ is constructed where each point $$\text{x}_i^{(l)}$$ is connected to $$k_l$$ other points through edges $$(i, j_{i1}) \ldots (i, j_{i k_l})$$. To perform this dynamic graph recomputation, the authors compute the pairwise distance between each point in the feature space $$\mathbb{R}^{F}$$, and the $$k_l$$ nearest neighbors are chosen for each point. With this graph constructed, the output of the actual EdgeConv operation is defined as the aggregation of weighted edge features for each edge in the neighborhood,

$$
\text{x}'_i =  \mathop{\square}_{j:(i,j)\in\mathcal{E}} h_{\Theta}(\text{x}_i, \text{x}_j).
$$

Here, $$\square$$ represents the chosen aggregation function, such as a summation, max, or average. The edge features $$h_{\Theta}(\text{x}_i, \text{x}_j)$$ have learnable paramters $$\Theta$$ and map from $$\mathbb{R^F} \times \mathbb{R^F}$$ to $$\mathbb{R}^{F'}$$. This operation is also shown in Fig 1.

![EdgeConv]({{ '/assets/images/team24/edge_pic.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. The EdgeConv operation* [1].

The choice of $$\square$$ and $$h$$ can greatly affect the performance of the model. For example, in the case where  

$$
h_{\Theta}(\text{x}_i, \text{x}_j) = h_{\Theta}(\text{x}_i).
$$

the features of the neighboring points are not taken into consideration when computing edge features. The authors choose to use an asymmetric function given by 

$$
h_{\Theta}(\text{x}_i, \text{x}_j) = h_{\Theta}(\text{x}_i, \text{x}_j - \text{x}_i).
$$

This choice of edge feature allows for exploiting both global structure as well as local neighborhood shape information. This operator is formulated as 

$$
h_{\Theta}(\text{x}_i, \text{x}_j)_m = \text{ReLU}(\phi_m \cdot \text{x}_i + \theta_m \cdot (\text{x}_j - \text{x}_i)),
$$

where $$\Theta = (\theta_1, \ldots, \theta_M, \phi_1, \ldots, \phi_M)$$ and the aggregation function is chosen as a maximum. This gives a final edge feature representation of 

$$
\text{x}'_{im} =  \mathop{\text{max}}_{j:(i,j)\in\mathcal{E}} h_{\Theta}(\text{x}_i, \text{x}_j)_m.
$$

In the original network architecture, shown in Fig 2, four EdgeConv layers are used to find geometric features from the input pointclouds. For each of these layers, the number of nearest neighbors $$k$$ is chosen as 20. The features from each layer are concatenated and passed through a global max pooling layer followed by two fully connected layers with a dropout of $$p=0.5$$. In addition, each layer uses a Leaky ReLU activation with batch normalization. Training is performed using SGD with momentum equal to 0.9, a learning rate of 0.1, and learning rate decay with cosine annealing [1].

![EdgeArch]({{ '/assets/images/team24/edgeconv.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. EdgeConv architecture for classification and segmentation* [1].

# 2D-3D Fusion

Our second network was also a GCN that relies on a backbone for feature extraction but differs in both the data flow and convolution operation. Instead of using a single graph network that takes in the points and associated features, we create two separate graphs, one representing 3D geometric features, and the second represending 2D texture features. The 2D texture features are extracted from the backbone and paired with a downsampled point cloud. The 3D geometric features, on the other hand, are solely based on the point cloud and depth data and are the result of using Attention Graph Convolution (AGC) [10] for both the euclidean and feature neighborhoods. This AGC operation is similar to EdgeConv with an attention mechanism except that AGC only includes edge attributes in its attention mechanism and not the features themselves. This is distinct from graph attention networks (GAT) [9] which takes a more common form of attention in which the attention mechanism takes into account the features of both nodes (not only the difference between the features) and also includes a normalization of the attention coefficient. We use an aggregation of a euclidean and feature-based AGC, which [11] calls MUNEGC and is seen below.

$$
\begin{equation}
\begin{array}{r}
\mathbf{x}_{i}^{\prime}=\Box\left\{\frac{1}{\left|N_{e}(i)\right|} \sum_{j \in N_{e}(i)} tanh(W_{e}^{l}\left(A_{i j}\right)) \mathbf{x}_{j}+b_{e}^{l},\right. \\
\left.\frac{1}{\left|N_{f}(i)\right|} \sum_{j \in N_{f}(i)} tanh(W_{f}^{l}\left(A_{i j}\right)) \mathbf{x}_{j}+b_{f}^{l}\right\}
\end{array}
  \label{eq:agc}
\end{equation}
$$

The pooling operation used alongside MUNEGC is a modification of traditional voxel pooling named Nearest Voxel Pooling (NVP) [11]. With standard voxel pooling, voxels of resolution $$r_p$$ are created and points from the point cloud inside each voxel are replaced with their centroid with a feature computed through either an average or maximum operation. NVP builds on this by then reassigning each point in the original point cloud to the nearest centroid, and grouping all points with the same centroid. Centroids without assigned points are removed, and each remaining centroid is given a new position equal to the average position of its group and feature equal to either the maximum or average of its group's features.


# GeoMat Dataset
To assess the performance of our DGCNN and fusion networks on point cloud classification, and specifically material prediction, we chose the GeoMat dataset since it includes both RGB and Sparse Depth unlike many other RGB-only material datasets [7]. In addition to the RGBD data, the GeoMat dataset provides camera intrinsics and extrinsics, calculated surface normals, as well as the position of the image patch in the larger image. We use the provided camera intrinsics and sparse depth to project the 2D image points into 3D space and generate a point cloud. 

We also pre-processed the image data to improve training by normalizing the RGB images by channel. We also augment the 2D image data by performing random flips and rotations. After projecting the depth map to a 3D point cloud, we normalize the positions of each point to the interval (-1, 1). When training to obtain the 3D geometric features, we augmented the point cloud and depth features by performing random dropout, randomly rotating about the z-axis between 0 and 180 degrees, and randomly flipping horizontally. In addition, we employ a random 3D cropping method proposed by [11], in which a random point is chosen from the point cloud and a cropping radius is found which includes only a specified fraction $$f$$ of the original number of points.

The dataset consists of a total of 19 different materials, with each category consisting of 400 training and 200 testing examples. Training examples from the Cement - Granular, Grass, and Brick classes along with their constructed 3D point clouds are shown in Fig 3. 

![Exam]({{ '/assets/images/team24/train_ex.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. GeoMat training images for Cement, Grass, and Brick, along with constructed pointclouds* [7].


# Implementation
Here we outline the more specific network architectures that were implemented. For training all of these networks, we used an RAdam optimizer with a learning rate of 0.001 and betas of 0.9 and 0.999 as a default. We also used cross-entropy loss with label-smoothing to reduce overconfidence of our models. We used the PyTorch Geometric framework since it provides common operations on graphs and allows for efficient dataloading, sampling, and training.

### Feature backbone

We utilized a convolutional network to generate features that serve as input to downstream networks. We chose ConvNext [12] given its state-of-the-art performance. We used pre-trained weights for $$\verb'convnext_large'$$ and finetuned the lower 3 convolutional blocks as well as the classification head. The convolutional layers have the following filter sizes and depths: $$\verb'192, 384, 768, 1536', \verb'3, 3, 27, 3'$$. Taking inspiration from [13], we pick the 3rd layer with dimensions $$\verb'512x14x14'$$ and perform max pooling for layers 1 & 2 with kernel sizes 4 & 2 respectively and interpolate the final layer, in order to match the dimensions of the 3rd layer. This produces a feature size of $$\verb'2880x14x14'$$. In order to reduce the dimensionality of these features, we then perform 2D convolution with a kernel size of 1 and output dimension of 32 filters.

### DeepTen
The baseline image-approach for comparison was based on Deep Texture Encoding Network (DeepTEN) [14], which uses dictionary learning and a residual encoding to learn domain-specific information. This encoding layer was specifically proposed for material recognition tasks and serves as a generalization of encoding schemes such as Fisher Vectors which showed the best performance in [7]. We used ConvNext [12] as our feature backbone, a dictionary with $$\verb'32x128'$$ codewords.

### DGCNN

All MLPs used a dropout of 0.8 and a leaky ReLu activation slope of -0.2 in our DGCNN networks. Each sample consisted of 1000 randomly sampled points, as opposed to the 10000 generated by the projection, in order to conserve GPU memory. We note that the MLP for each EdgeConv layer must take in double the dimensions of the layer input.

$$\textit{DG-V1}: k=40$$, 3 EdgeConv layers $$\verb'[6, 64],[128,128],[256,256]'$$ and 2 MLP layers $$\verb'[448, 1024],[1024, 512, 256, 19]'$$. 

$$\textit{DG-V2}: k=20$$, 4 EdgeConv layers $$\verb'[6, 64],[128,64],[128, 128],[256,256]'$$ and 2 MLP layers $$\verb'[512, 1024],[1024, 512, 256, 19]'$$. 

$$\textit{DG-V3}$$: 1 2D Conv for the feature backbone: $$\verb'[1344, 128]'$$, 4 EdgeConv layers $$\verb'[268, 64],[128,128],[256,256]'$$ and 2 MLP layers $$\verb'[576, 1024],[1024, 512, 256, 19]'$$. 

$$\textit{DG-V4}$$: 1 2D Conv for the feature backbone: $$\verb'[64]'$$, 4 EdgeConv layers $$\verb'[6, 64],[128,64]'$$ and 2 MLP layers $$\verb'[192, 1024],[1024, 512, 256, 19]'$$.

We see a simplified version of $$\textit{DG-V4}$$ below:

```
class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr="max", dropout=0.8):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * (3 + 3), 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=dropout), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=dropout), k, aggr)
        self.fc1 = MLP([64 + 64, 1024], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=dropout)
        self.fc2 = MLP([1024 + 2304, 512, 256, out_channels], dropout=dropout)

        self.img_model = timm.create_model("convnext_base", num_classes=2, drop_path_rate=dropout).cuda()
        self.img_model.eval() # Don't finetune layers to reduce computation
        self.filter_conv = nn.Conv2d(1920, 64, 1)  # reduce filter size

    def forward(self, data):
        pos, x, batch = (data.pos.cuda(), data.x.cuda(), data.batch.cuda())
        features = self.img_model.get_features_concat(data.image.cuda().permute(0, 3, 1, 2).float())
        features = self.filter_conv(features)
        x1 = self.conv1(torch.cat((pos, x), dim=1).float(), batch)
        x2 = self.conv2(x1, batch)
        out = self.fc1(torch.cat((x1, x2), dim=1))
        out = global_max_pool(out, batch)
        out = self.fc2(torch.cat((out, features.reshape(features.shape[0], -1)), dim=1))
        return F.log_softmax(out, dim=1)
```

We also show the implementation of the EdgeConv operator (seen above as `DynamicEdgeConv`). The core of the code is the forward method where the KNN function is called and the message function where x_i and x_i - x_j are passed through the specified nonlinear function (in our case an MLP):

```
class DynamicEdgeConv(MessagePassing):
    ...
    def forward(self, x, batch):
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        ...

        b = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
```

### 2D-3D Fusion
The fusion approach uses 3 separately trained networks, two of which are used to extract 2D and 3D features from the input data, and one which fuses the 2D and 3D features for the final classification.

#### 2D Features
We use the same ConvNext [12] feature backbone for our texture graph and perform avg 2D pooling to reduce our point lattice to match the feature size $$(\verb'14x14')$$. 

#### 3D Features
The 3D geometric features are generated with a GCN which makes use of the MUNEGC and NVP layers described previously. The main building block consists of a MUNEGC layer followed by batch normalization, ReLU activation, and NVP pooling with maximum aggregation. The network uses 4 of these blocks, with output feature dimensions and pooling radius of $$\verb'16'$$ and $$\verb'0.05'$$, $$\verb'16'$$ and $$\verb'0.08'$$, $$\verb'32'$$ and $$\verb'0.12'$$, and $$\verb'64' and \verb'0.24'$$, respectively. This is then followed by another MUNEGC layer with output size $$\verb'128'$$ with batch normalization and ReLU activation. For training, a classification network also follows, with average global pooling, dropout of 0.2, and a fully connected layer with an output size of 19. After completing training, the output after the 5th MUNEGC layer is extracted to obtain the 3D geometric features which have a feature size of $$\verb'128'$$. Partial implementation of the AGC operator is shown below, courtesy of [10]. 

```
class AGC(torch.nn.Module):
    ...
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)

        src_indx = edge_index[j]
        target_indx = edge_index[i]

        # weights computation
        out = self._wg(edge_attr)
        out = out.view(-1, self.in_channels, self.out_channels)

        N, C = x.size()

        feat = x[src_indx]

        out = torch.matmul(feat.unsqueeze(1), out).squeeze(1)
        out = scatter(out, target_indx, dim=0, dim_size=N, reduce=self.aggr)

        if self.bias is not None:
            out = out + self.bias

        return out
```

#### Fusion
The geometric and texture feature networks are trained individually and their weights are then frozen. We then implement the "Multi-model group fusion" proposed in [11]. In our case, we have the geometric point features with $$\verb'dim 128'$$ and the texture features with $$\verb'dim 1920'$$, which are first projected to 3D space. We then pass both sets of features through a ReLu activation before performing 1D convolution to match their dimensionality to $$\verb'dim 512'$$. This is the first step to fusing the features. These features are then fused using a modified version of the Nearest Voxel Pooling used in obtaining the 3D geometric features. Initially, points from the 2D and 3D features are assigned to a centroid in the same 2-step procedure used in NVP and centroids without assigned points are removed. Next, for a given centroid $$c_i$$, the averages of all the assigned 3D and 2D features points are calculated separately, and then the two averages are concatenated to create the final feature. In the case that a centroid has only one type of feature point, the missing feature average is replaced with all ones. The location of the new centroid is chosen as the average of all the 2D and 3D feature points in the group. The network consists of this NVP layer with a radius of 0.24, followed by a classification network with batch normalization, ReLU activation, global average pooling, dropout of 0.5, and a final fully connected layer with output size 19. We show part of the fusion network below:

```
class MultiModalGroupFusion(torch.nn.Module):
    ...
    def forward(self, b1, b2):
        pos = torch.cat([b1.pos, b2.pos], 0)
        batch = torch.cat([b1.batch, b2.batch], 0)

        batch, sorted_indx = torch.sort(batch)
        inv_indx = torch.argsort(sorted_indx)
        pos = pos[sorted_indx, :]

        start = pos.min(dim=0)[0] - self.pool_rad * 0.5
        end = pos.max(dim=0)[0] + self.pool_rad * 0.5

        cluster = torch_geometric.nn.voxel_grid(pos, self.pool_rad, batch, start=start, end=end)
        cluster, perm = consecutive_cluster(cluster)

        superpoint = scatter(pos, cluster, dim=0, reduce="mean")
        new_batch = batch[perm]

        cluster = nearest(pos, superpoint, batch, new_batch)

        cluster, perm = consecutive_cluster(cluster)

        pos = scatter(pos, cluster, dim=0, reduce="mean")
        branch_mask = torch.zeros(batch.size(0)).bool()
        branch_mask[0 : b1.batch.size(0)] = 1

        cluster = cluster[inv_indx]

        nVoxels = len(cluster.unique())

        x_b1 = torch.ones(nVoxels, b1.x.shape[1], device=b1.x.device)
        x_b2 = torch.ones(nVoxels, b2.x.shape[1], device=b2.x.device)

        x_b1 = scatter(b1.x, cluster[branch_mask], dim=0, out=x_b1, reduce="mean")
        x_b2 = scatter(b2.x, cluster[~branch_mask], dim=0, out=x_b2, reduce="mean")

        x = torch.cat([x_b1, x_b2], 1)
        ...
```

# Results
The results of our experiments are shared here. Training curves are shown in \ref{asf} confusion matrices on the test set are shown in \ref{confusion}.

### DeepTen
With the DeepTen network structure consisting of our feature backbone, encoding layer, and fully connected layer, we are able to achieve $$\textbf{68.55%}$$ accuracy on the test set.

### DGCNN
With our DGCNN using only depth data, we are able to achieve $$\textbf{38.00%}$$ accuracy on the test set $$(\textit{DG-V1})$$. Adding RGB channels as node features dramatically improves our results, with $$\textbf{61.45%}$$ test accuracy $$(\textit{DG-V2})$$. Our best result is achieved using our feature backbone in addition to the RGBD features per node, and a 4 layer network with EdgeConv with 2 following fully connected layers. This results in a test accuracy of $$\textbf{76.55%}$$ $$(\textit{DG-V3})$$.

We also develop a much lighter version of our DGCNN that uses just two EdgeConv layers followed by two fully-connected layers, and a lighter feature backbone, $$\verb'convnext_base'$$. This results in 89 million parameters, including the backbone, compared to the 198 million parameters in $$\textit{DG-V3}$$. For this model, we only pass the image features to the fully connected layer, and bypass the EdgeConv layers to reduce computation. We achieve $$\textbf{75.08%}$$ accuracy on the test set $$(\textit{DG-V4})$$, confirming our assumption that the graph convolution layers are most effective at processing the raw pixel-wise data and not the preprocessed and interpolated image features since these lack meaningful structure.

### Fusion
Our fusion network combining 2D image and 3D depth features achieves $$\textbf{77.21%}$$ accuracy on the test set, outperforming the best result of 73.84% from [7] by more than 3%. We can see from the confusion matrix \ref{confusion} that the most common errors occur between the asphalt, cement, and concrete classes of which there are 5 in total.

RESULTS TABLE

CONFUSION MATRICES

TRAINING CURVES


# Discussion

However, we iteratively improved our DGCNN by adding image features, fine-tuning the backbone and increasing the KNN param, all of which contribute to the improved results seen in Table \ref{table}.

We also see that the geometric features are only able to achieve 33.66% accuracy on the test set when not augmented with image features. We attribute this result to both the dataset type and resolution of the depth data. As there are several material types in the dataset with similar physical geometry, it is likely that the sparse depth data is simply insufficient for classification. It is possible that utilizing the contextual depth data (e.g. the rest of the scene) may improve the geometric classification results similar to how contextual encoding of image features has been shown to improve RGB-only semantic segmentation [15]. Nonetheless, we do see an improvement with both the DGCNN and the fusion networks compared to the image-only DeepTen model. 

We also see that the fusion network that utilizes attention graph convolution (AGC) outperforms our DGCNN with EdgeConv by 0.66% in accuracy on the test set. However, due to the differences in the respective networks, it is hard to draw a direct comparison. We do, however, see evidence that incorporating the image features after the graph convolution steps, is critical to retaining the texture information. Both the fusion network and the successful DGCNN networks $$(\textit{DG-V3}, \textit{DG-V4})$$ either separated the features entirely, or added a skip connection over the graph convolution layers. It is clear between the performance of $$\textit{DG-V2}$$ and $$\textit{DG-V3}$$ that this skip connection leads to a large performance improvement.



## Reference

[1] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph CNN for learning on point clouds,” ACM Transactions on Graphics, vol. 38, no. 5, pp. 1–12, 2019.

[2] X. Wei, R. Yu, and J. Sun, “View-GCN: View-based graph convolutional network for 3D shape analysis,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[3] Q. Xu, X. Sun, C.-Y. Wu, P. Wang, and U. Neumann, “Grid-GCN for fast and Scalable Point Cloud Learning,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[4] Y. Chai, P. Sun, J. Ngiam, W. Wang, B. Caine, V. Vasudevan, X. Zhang, and D. Anguelov, “To the point: Efficient 3D object detection in the range image with graph convolution kernels,” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[5] H. Haotian,  F. Wang and H. Le. “VA-GCN: A Vector Attention Graph Convolution Network for learning on Point Clouds.” ArXiv abs/2106.00227 2021.

[6] L. Chen and Q. Zhang, “DDGCN: Graph convolution network based on direction and distance for point cloud learning,” The Visual Computer, 2022.

[7] J. DeGol, M. Golparvar-Fard, and D. Hoiem, “Geometry-Informed Material Recognition,” 2016 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[8] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” in Proc. IEEE CVPR, Jul. 2017

[9] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

[10] A. Mosella-Montoro and J. Ruiz-Hidalgo. Residual attention graph convolutional network for geometric 3d scene classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019.

[11] A. Mosella-Montoro and J. Ruiz-Hidalgo. 2d–3d geometric fusion network using multi-neighbourhood graph convolution for rgb-d indoor scene classification. Information Fusion, 76:46–54, 2021.

[12] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. arXiv preprint arXiv:2201.03545, 2022.

[13] S. Casas, A. Sadat, and R. Urtasun. Mp3: A unified model to map, perceive, predict and plan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14403–14412, 2021.

[14] H. Zhang, J. Xue, and K. Dana. Deep ten: Texture encoding network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 708–717, 2017.

[15] H. Zhang, K. Dana, J. Shi, Z. Zhang, X. Wang, A. Tyagi, and A. Agrawal. Context encoding for semantic segmentation. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 7151–7160, 2018.

## Code Repository
[1] [Dynamic Graph CNN for Learning on Point Clouds](https://github.com/WangYueFt/dgcnn)

[2] [Pytorch code for view-GCN](https://github.com/weixmath/view-GCN)

[3] [Grid-GCN for Fast and Scalable Point Cloud Learning](https://github.com/Xharlie/Grid-GCN)

[4] [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)

---
