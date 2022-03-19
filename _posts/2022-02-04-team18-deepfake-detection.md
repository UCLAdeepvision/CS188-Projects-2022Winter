---
layout: post
comments: true
title: Proposal

author: UCLAdeepvision
date: 2022-02-04
---

> 


<!--more-->
{: class="table-of-content"}
* Introduction
* Deepfake Generation
* Deepfake Datasets
* MesoNet
* Experiments
* Data Augmentation
* Takeaways
* Demo
* References
{:toc}

## Introduction

With the rise of portable smartphone camera technology and the growth of digital media, content has transitioned from being primarily text-based to being centered around images and videos. With this has come increasingly sophisticated techniques to forge digital content that is indiscernable from authentic content, also known as a deepfake. There are several harmless, and even fun, applications for deepfakes, but they can also be used detrimentally to generate fake political or social content, creating a need to be able to effectively detect fake content. 

Both creating and detecting deepfakes is still an active area of research, but several advanced techniques have been developed that allow for the generation of high-quality deepfakes. The content produced by these methods is still lossy and may contain artifacts that, while not visible to the human eye, seperate it from an authentic camera image. In this article, we try to improve on the performance of existing models that attempt to characterize and identify these faults in deepfakes in order to classify them. [3]

## Deepfake Generation

To truly understand deepfake detection, we must first understand how they are generated. The most common technique is to use paired autoencoders. Autoencoders contain two stages, the encoder and the decoder. The encoder tries to reduce the dimensionality of the image such that the most pertinent features are preserved, while the decoder attempts to use these features to recreate the input image. The generated image is then compared to the input image to compute the L2 loss, which is backpropogated to train both the decoder and the encoder. 

To generate a deepfake, we start with two sets of input images, A and B, with pictures of two different people whose faces we want to swap. We then train the autoencoders E<sub>A</sub> and E<sub>B</sub> on their respective set of images. Both autoencoders share the same encoder, but have seperate decoders. This allows the encoder to capture the most general information about a face, such as position and orientation, while the decoder can reconstruct the face given certain unique characteristics. When implemented correctly, this technique can produce high-quality results that often cannot be easily discerned from a real image. [3]

## Deepfake Datasets 

### FaceForensics++

The FaceForesics++ dataset is a compilation of over 1.8 million images sampled from over 1,000 videos. The dataset was developed by the Google AI team and is available for free to researchers around the world. Each deepfake is paired to a pristine source video and ground-truth labels are avialable to allows for supervised learning. Several methods were employed to develop the dataset, including but not limited to FaceSwap, Face2Face, and NeuralTextures. The Google AI team has also provided several models that have been pretrained on this dataset. [1]

### Deepfake Detection Challenge Dataset

The Deepfake Detection Challenge Dataset, developed by Facebook AI, is a collection of over 100,000 clips sourced from over 3,000 paid actors. The deepfakes were generated using a variety of methods, inlcuding but not limited to DFAE, FSGAN, and StyleGAN. The dataset was used for the public Deepfake Detection Challenge hosted on Kaggle in collaboration with AWS, Facebook, Microsoft, and Partnership on AI’s Media Integrity Steering Committee. Each clip represents a unique face swap with varying quality, allowing models trained on this dataset to be robust and scalable. Ground-truth labels are available for all training data. [2]

## MesoNet

MesoNet is a set of architextures proposed by Afchar et. al with several properties that make it especially well suited for Deepfake Detection applications. For our applications, we made use of the Meso-4 architexture, which stacks four convolutional layers, followed by two fully-connected layers for classification. Each convuluational layer is followed by a batch normalization layer and a 4x4 max pooling layer. Furthermore, the model employs dropout regularization in the fully-connected layers for robustness during classification. 

Since the deepfake generation process produces considerable noise, a microscopic apporach to feature extraction tends to be ineffective. However, since the images are of high-enough quality to be indistinguishable to the human eye, looking only at high-level features fails to provide a good mechanism for classifiying deepfakes. The moderate size of Meso-4 allows it to look at the intermediate-level (or mesoscopic) features of the images. Furthermore, the low number of trainable features makes both training and evaluating the model relatively inexpensive. The researchers have provided models pretrained on the FaceForensics++ dataset [3]. 


### Experiments

We took a transfer learning approach to the deepfake detection task. Namely, we utilize the pretrained weights from the MesoNet architecture on the FaceForesnics++ Dataset, allowing us to get the best of both worlds, with both an architecture and a dataset that perform extremely well in Deepfake Detection. The researchers who devised the FaceForensics++ dataset devised their own benchmark, XceptionNet, to test the performance of the dataset. This consists of a basic CNN architecutre that was trained on ImageNet for 3 epochs, before final layer is trained on the highest performing model on the validation set, for a total of 15 epochs (FACE FORENSICS) . Table 1 compares the accuracy of the XceptionNet trained on the FaceForensics++ dataset with the MesoNet architecture, considered state of the art for the task at hand. 

| Model (15 Epochs)          |      Accuracy     |
| :------------------------- | :---------------: |
| MesoNet                    |       0.87        |
| XceptionNet                | $$\textbf{0.96}$$ |


_Table 1._

We can see that despite using a traditional CNN architecture, the model trained on the FaceForensics++ dataset heavily outperforms state of the art architecture, communicating the importance of high quality data in deepfake detection. Because of the computational costs associated with video processing, we utilized a small subset of around 1000 videos. Pre processing code adopted from MesoNet was utilized to split the videos by frames into a series of images, and isolate the faces from the images (MESONET). Then, we utilized data augmentation to increase the size of our data with minimal computational cost, while allowing the architecture to adapt to different variations of images and video. We can see the Data Augmentation code below.

### Data Augmentation

```py
#Transformations for data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=120,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.05,
    shear_range=120,
    zoom_range=0.9
)

#Fit classifier with augmented data
classifier.model.fit(
    datagen.flow(X_train, y_train, batch_size=32, subset='training'),
    validation_data = datagen.flow(X_train, y_train, batch_size=8, subset='validation'),
    epochs = 100
)

#Evaluate model
loss = classifier.model.evaluate(
    x = X_test,
    y = y_test
)
```

Additionally, we implemented layer locking to ensure the integrity of the pretrained weights from MesoNet. Layers outside of the dense were locked. The utilization of data augmentation and transfer learning on the pretrained MesoNet weights is where the innovation lies in our deepfake detection task. We trained this architecture on 15 epochs like our benchmarks. 

| Epochs                     |      Accuracy     |
| :------------------------- | :---------------: |
| 0 Epochs (Pretrained weights)|       0.93      |
| 15 Epochs (Pretrained weights)|       0.93      |
| 15 Epochs (No pretrained weights)|     0.06     |
| 100 Epochs (Pretrained weights)|      0.93       |

_Table 2._

Table 2 above displays the results of these experiments. The accuracy was 0.93 which was the same  accuracy that the pre trained weights from MesoNet achieved without training on the FaceForensics++ dataset. This shows that the pretrained MesoNet architecture does perform extremely well in deepfake detection as is. When reaching such accuracy as 93%, even a 0.5% boost in accuracy would be a major accomplishment, which is why methods such as transfer learning and data augmentation were applied. 

### Takeaways

We can see that applying transfer learning to the FaceForensics++ dataset had minimal impact on the performance of our model. The MesoNet paper gave us a state of the art architecture to detect deepfakes, utilizing information about the way they were generated (via GANs) to identify noise in image data. Training further to even 100 epochs did not improve the accuracy. It is clear then that there is little information that the model is able to learn beyond the information it already has from the pretrained weights of MesoNet. One reasonable conclusion for this and a pathway for future work is that the 1000 video subset that we used was simply not sufficient. Future work could include utilizing a much larger percentage (we used about 1% due to limitations in storage and computational capacity) of the dataset when training. This will require a lot of time and/or computational power, but methods to optimize this process can be a part of this future work. See below for a demo.

### Demo

A link to a Colab demo can be found here: https://colab.research.google.com/drive/1MicgU8F5lmu-LwB54BWma60mmcr_cFNW?usp=sharing

A repository containing code to train and evaluate the model can be found here: https://github.com/arnavgarg/deepfake-detection-demo

## References

[1] Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner: “FaceForensics++: Learning to Detect Manipulated Facial Images”

[2] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin Wang, Cristian Canton Ferrer: “The DeepFake Detection Challenge (DFDC) Dataset”, 2020

[3] Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen: “MesoNet: a Compact Facial Video Forgery Detection Network”, 2018;