---
layout: post
comments: true
title: An Introduction to Medical Image Segmentation
author: Aaron Minkov
date: 2022-02-28
---


> Medical image segmentation serves as the backbone of medical image processing in today's world. In order to account for the variability in medical imaging, medical image segmentation detects boundaries within 2D and 3D images in order to identify crucial features and sizes of objects within them. This has tremendously assisted research, diagnosis, and computer-based surgery within the medical field. With the rise of deep learning algorithms, medical image segmentation has seen an increase in accuracy and performance and has led to incredible new innovations within the medical field.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
The rise of deep learning in recent years has allowed many fields of AI to flourish. Medical image segmentation existed before deep learning was around, but the introduction of deep learning led the resourcefulness and accuracy of such segmentation to flourish. There have been several main issues with medical image segmentation that previously limited the process' abilities: variability within medical images, variability within human tissue, noise between image pixels, and inherent uncertainty due to limitations of knowledge within the medical world. Though these issues may remain indefinitely, deep learning has allowed the process of image segmentation to achieve better results than ever, and its potential is far more promising than the algorithms that came before it.

## What is Medical Image Segmentation?
Medical image segmentation is the process of identifying main features within medical images by labeling each pixel with an object class such as 'heart', 'tumor', 'artery', etc. With deep learning, we are able to train models in order to automatically assign labels to pixels with high accuracy. These advancements have allowed the performance of automatic image segmentation to match that of professionally trained radiologists [2].

![SegmentationExample]({{ '/assets/images/team05/figure1.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. An example of manual & automatic segmentation on lesions (green) within a liver (red)* [4].

As seen in Figure 1, the outputs of both manual and automatic segmentations of a liver with lesions are quite similar. And of course, the benefits of automatically segmenting these medical screenings include quicker interpretation times, less room for user error, and a second layer of analysis (first via the deep learning model, and second by a medical professional to confirm the accuracy of a scan's results).

## U-Net
U-Net is a type of convolutional neural network (CNN) used specifically for medical image segmentation. It is a very popular CNN due to its ability to train with fewer samples while still producing more accurate segmentations on testing data. It utilizes four encoders and four decoders which generates a U-shaped architecure, hence why the network got its name.

![UNetArchitecture]({{ '/assets/images/team05/figure2.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. U-Net architecture consisting of four decoders, four encoders, and a bridge to connect the two. The blue boxes are feature maps, and the channel counts for each are listed above each box* [5].

A simple implementation of what Figure 2 illustrates can be seen in [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), a Python framework developed by the GitHub user [milesial](https://github.com/milesial). It utilizes PyTorch to create encoder and decoder functions that process the features within each image and ultimately outputs a segmentation map which attempts to identify important characteristics within each image.

### Encoding in U-Net
Encoders are networks that take an input and convert it to a feature map that identifies various important features within the original input. U-Net utilizes upsampling as its form of encoding which means each layer of encoding increases the output resolution. This allows high-resolution features to map more precisely following each convolution layer.

Within each encoder, we have two 3x3 convolutions followed by a Rectified Linear Unit (ReLU) activation function which helps prevent linearity as the model analyzes training data. After each block, we use a 2x2 max-pooling function which reduces the height & width of the feature map by 1/2 in order to maximize efficiency and minimize the cost of computation.

### Decoding in U-Net
TODO

## Implementation via MIScnn
MIScnn is an open-source Python framework that allows you to implement medical image segmentation utilizing deep learning models and convolutional neural networks. Using this API, you can set up a segmentation model with just a few lines of code.

### Setup
```py
# Import necessary modules
import miscnn
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn.neural_network.architecture.unet.standard import Architecture

# Create a Data I/O interface (in this case we
# use one for kidney tumor CT scans in NIfTI format)
interface = NIFTI_interface(pattern="case_000[0-9]*", channels=1, classes=3)

# Initialize data path and create the Data I/O instance
data_path = "<YOUR PATH HERE>"
io = miscnn.Data_IO(interface, data_path)

# Create a Preprocessor instance to configure how to preprocess the data into batches
pre_processor = miscnn.Preprocessor(io, batch_size=4, analysis="patchwise-crop",
                         patch_shape=(128,128,128))

# Create a deep learning neural network model with a standard U-Net architecture
unet_standard = Architecture()
model = miscnn.Neural_Network(preprocessor=pre_processor, architecture=unet_standard)
```

Now that you've set up your model, you're ready to train it based on the sample data you've provided. Yes, with MIScnn it's <b>that simple</b>.

### Training Your Model
```py
# Train your model using 80 samples from your training data, and run it for 50 epochs
# You can increase or decrease the # of epochs depending on your GPU power
samples = io.get_indiceslist()
model.train(samples[0:80], epochs=50)

# Now, we predict the segmentation for 20 samples
preds = model.predict(samples[80:100], return_output=True)
```

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [4].

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

## Reference

[1] Withey, D. J. & Koles, Z. J. A review of medical image segmentation: methods and available software. Int. J. Bioelectromagn.10, 125–148 (2008).<br>
[2] Müller, D., Kramer, F. MIScnn: a framework for medical image segmentation with convolutional neural networks and deep learning. BMC Med Imaging 21, 12 (2021). https://doi.org/10.1186/s12880-020-00543-7<br>
[3] Liu, X.; Song, L.; Liu, S.; Zhang, Y. A Review of Deep-Learning-Based Medical Image Segmentation Methods. Sustainability 2021, 13, 1224. https://doi.org/10.3390/su13031224<br>
[4] Vorontsov, Eugene & Tang, An & Pal, Chris & Kadoury, Samuel. (2018). Liver lesion segmentation informed by joint liver segmentation. 1332-1335. 10.1109/ISBI.2018.8363817.<br>
[5] Olaf Ronneberger, Philipp Fischer, Thomas Brox. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation

---
