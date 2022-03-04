---
layout: post
comments: true
title: Sheet Music Recognition
author: Ning Wang and Alan Yao
date: 2022-01-26
---


> Sheet Music Recognition is a difficult task. [Zaragoza et al.](URL 'https://www.mdpi.com/2076-3417/8/4/606') devised a method for recognizing monophonic scores (one staff). We extend this functionality for piano sheet music (grand staff) that have are monophonic in each staff (treble and bass).


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
This project was inspired by [Zaragoza et al.](URL 'https://www.mdpi.com/2076-3417/8/4/606'). We extend the monophonic score reader by parsing grand staves from piano sheet music. Thus, we add a stage in the pipeline to first identify any grand staves before separating them into treble and bass. Each individual staff is then feed into the current pipeline. 


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

## Reference
[1] Calvo-Zaragoza, J., Rizo, D.: End-to-end neural optical music recognition of monophonic scores. Appl. Sci. 8(4), 606 (2018)

[2] Calvo-Zaragoza, J., Rizo, D.: Camera-PrIMuS: neural end-to-end optical music recognition on realistic monophonic scores. In: Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, 23–27 September 2018, pp. 248–255 (2018)

[3] Alfaro-Contreras M., Calvo-Zaragoza J., Iñesta J.M. (2019) Approaching End-to-End Optical Music Recognition for Homophonic Scores. In: Morales A., Fierrez J., Sánchez J., Ribeiro B. (eds) Pattern Recognition and Image Analysis. IbPRIA 2019. Lecture Notes in Computer Science, vol 11868. Springer, Cham.


---
