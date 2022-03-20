---
layout: post
comments: true
title: Enhanced Self-Driving with combination of map and lidar inputs.
author: Alexander Swerdlow, Puneet Nayyar
date: 2022-01-27
---


> 3D Point Cloud understanding is critical for many robotics applications with unstructured environments such as self-driving cars. LIDAR sensors provide accurate, high-resolution point clouds that provide a clear view of the environment but making sense of this data can be computationally expensive and is generally difficult. Graph convolutional networks aim to exploit geometric information in the scene that is difficult for 2D based image recognition approaches to reason about.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Our project seeks to implement and improve upon graph convolution approaches to 3D point cloud classification and segmentation. We plan to use the dynamic graph approach proposed by [1] which creates a graph between each layer with connections between nearby points and uses an asymmetric edge kernel that incorporates relative and absolute vertex locations. This approach contrasts with the typical approach to learning on point clouds which operates directly on sets of points. PointNet [8] is one popular implementation that takes this approach, learning spatial features for each point and using pooling to later perform classification or segmentation.

## Introduction To Graph Neural Networks

Graph Neural Networks, or GNNs, are as the name suggests, neural networks that make sense of graph informaton. This umbrella term covers graph-level tasks (which our project focuses on), node-level tasks (e.g. predicting the property of a node), and edge-level tasks. GNNs take some input graph $$G = (V, E)$$ and transform that graph in some sequence to produce a desired prediction. A basic GNN might look like several layers, each composed of some non-linear function $$f$$ (e.g. an MLP) that takes in the current graph $$\mathcal{G}_i$$. Depending on the GNN, the graph structure, commonly represented by an adjacency matrix, may stay the same throughout the layers, or change at each layer. The node and/or edge features are progessively transformed by this non-linear mapping and, if the task is graph-level classification, some pooling operation is typically performed to reduce the dimensionality.Depending on the GNN, the graph structure, commonly represented by an adjacency matrix, may stay the same throughout the layers, or change at each layer.

## EdgeConv
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

For the final network architecture, shown in Fig 2, four EdgeConv layers are used to find geometric features from the input pointclouds. For each of these layers, the number of nearest neighbors $$k$$ is chosen as 20. The features from each layer are concatenated and passed through a global max pooling layer followed by two fully connected layers with a dropout of $$p=0.5$$. In addition, each layer uses a Leaky ReLU activation with batch normalization. Training is performed using SGD with momentum equal to 0.9, a learning rate of 0.1, and learning rate decay with cosine annealing [1].

![EdgeArch]({{ '/assets/images/team24/edgeconv.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. EdgeConv architecture for classification and segmentation* [1].


## GeoMat Dataset
To assess the performance of EdgeConv on point cloud classification, and specifically material prediction, we chose the GeoMat dataset since it includes both RGB and Sparse Depth unlike many other RGB-only material datasets [7]. In addition to the RGBD data, the GeoMat dataset provides camera intrinsics and extrinsics, calculated surface normals, as well as the position of the image patch in the larger image. We use the provided camera intrinsics and sparse depth to project the 2D image points into 3D space and generate a point cloud. 

```
from torch_geometric.data import (Data, Dataset)

class GeoMat(Dataset):
    def __init__(self, root, train=True, transform=None,
                pre_transform=None, pre_filter=None):

        self.train_raw = self.read_txt(osp.join(root, 'raw_train.txt'))
        self.test_raw = self.read_txt(osp.join(root, 'raw_test.txt'))
        self.train_proc = self.read_txt(osp.join(root, 'processed_train.txt'))
        self.test_proc = self.read_txt(osp.join(root, 'processed_test.txt'))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.train = train
        self.data = self.train_proc if self.train else self.test_proc
    ... 
    def process(self):
        raw_filenames = self.raw_paths
        processed_filenames = self.processed_paths
        for raw_fn, proc_fn in zip(raw_filenames, processed_filenames):
            f = sio.loadmat(raw_fn)
            label = torch.from_numpy(f['Class'][0]).to(torch.long) - 1 
            depth = np.ascontiguousarray(f['Depth'].astype(np.float32))
            rgb = np.ascontiguousarray(f['Image']) 
            intrinsics = f['Intrinsics'].astype(np.float64)
            extrinsics = np.vstack([f['Extrinsics'].astype(np.float64), [0, 0, 0, 1]])

            im_rgb, im_depth = o3d.geometry.Image(rgb), o3d.geometry.Image(-depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, 
                                                                      im_depth, 
                                                                      convert_rgb_to_intensity=False)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(rgb.shape[1], 
                                                           rgb.shape[0], 
                                                           intrinsics[0,0], 
                                                           intrinsics[1,1], 
                                                           intrinsics[0,2], 
                                                           intrinsics[1,2])
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                                                 intrinsics, 
                                                                 extrinsics, 
                                                                 project_valid_depth_only=True)

            pointcloud = torch.from_numpy(np.asarray(pcd.points))
            pointcloud_rgb = torch.from_numpy(np.asarray(pcd.colors))
            data = Data(pos=pointcloud, x=pointcloud_rgb, y=label)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            os.makedirs(osp.dirname(proc_fn), exist_ok=True)
            torch.save(data, proc_fn)
```

The dataset consists of a total of 19 different materials, with each category consisting of 400 training and 200 testing examples. Training examples from the Cement - Granular, Grass, and Brick classes along with their constructed 3D point clouds are shown in Fig 3. 

![Exam]({{ '/assets/images/team24/train_ex.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. GeoMat training images for Cement, Grass, and Brick, along with constrcuted pointclouds* [7].


## Implementation

We chose 4 EdgeConv layers as in [1] with single linear layers between the convolutions. Each linear layer has batch normalization and we used Leaky ReLu as our activation function. We used an Adam optimizer with learning rate equal to 0.001 and decay of 0.5 every 20 epochs. We use PyTorch-Geometric [5] since it provides common operations on graphs and allows for efficient dataloading, sampling, and training.

```
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr="max"):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64], 
            act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64], 
            act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128], 
            act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256], 
            act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.fc1 = MLP([64 + 64 + 128 + 256, 1024], 
            act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5)
        self.fc2 = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        x1 = self.conv1(torch.cat((pos, x), dim=1).float(), batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.fc1(torch.cat((x1, x2, x3, x4), dim=1))
        out = global_max_pool(out, batch)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

def train():
    model.train()
    train_loss, train_pred, train_true = 0, [], []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        preds = out.max(dim=1)[1]
        train_loss += loss.item() * data.num_graphs
        train_true.append(data.y.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return train_loss / len(train_dataset), 
        metrics.accuracy_score(train_true, train_pred), 
        metrics.balanced_accuracy_score(train_true, train_pred)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)  # T.SamplePoints(1024)
train_dataset = GeoMat(path, True, transform, pre_transform)
test_dataset = GeoMat(path, False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=6)
model = Net(in_channels=6, out_channels=19, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
```

## Results
Preliminary results are shown here. 

![TrainLoss]({{ '/assets/images/team24/Loss_train.svg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 5. Training loss vs. Epoch*.

![TrajnAcc]({{ '/assets/images/team24/Accuracy_train.svg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 6. Training accuracy vs. Epoch*.

![TrainAccB]({{ '/assets/images/team24/Balanced_Accuracy_train.svg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 7. Balanced training accuracy vs. Epoch*.

![TestAcc]({{ '/assets/images/team24/Accuracy_test.svg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 8. Test accuracy vs. Epoch*.

## Reference

[1] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph CNN for learning on point clouds,” ACM Transactions on Graphics, vol. 38, no. 5, pp. 1–12, 2019.

[2] X. Wei, R. Yu, and J. Sun, “View-GCN: View-based graph convolutional network for 3D shape analysis,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[3] Q. Xu, X. Sun, C.-Y. Wu, P. Wang, and U. Neumann, “Grid-GCN for fast and Scalable Point Cloud Learning,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[4] Y. Chai, P. Sun, J. Ngiam, W. Wang, B. Caine, V. Vasudevan, X. Zhang, and D. Anguelov, “To the point: Efficient 3D object detection in the range image with graph convolution kernels,” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[5] H. Haotian,  F. Wang and H. Le. “VA-GCN: A Vector Attention Graph Convolution Network for learning on Point Clouds.” ArXiv abs/2106.00227 2021.

[6] L. Chen and Q. Zhang, “DDGCN: Graph convolution network based on direction and distance for point cloud learning,” The Visual Computer, 2022.

[7] J. DeGol, M. Golparvar-Fard, and D. Hoiem, “Geometry-Informed Material Recognition,” 2016 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[8] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” in Proc. IEEE CVPR, Jul. 2017


## Code Repository
[1] [Dynamic Graph CNN for Learning on Point Clouds](https://github.com/WangYueFt/dgcnn)

[2] [Pytorch code for view-GCN](https://github.com/weixmath/view-GCN)

[3] [Grid-GCN for Fast and Scalable Point Cloud Learning](https://github.com/Xharlie/Grid-GCN)

[4] [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)

---
