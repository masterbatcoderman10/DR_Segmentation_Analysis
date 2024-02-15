This folder contains the code for the submission. The contents of this folder should be:
- annotations-color.zip : 8 combined color annotation maps
- annotations-folder.zip : 8 individual annotations folders for each fundus image
- images.zip : 8 fundus images
- Predictions.ipynb : This file contains the code to be run.

The Predictions.ipynb file is where all the code for this demonstration resides, the zip files are the associated data that is required. Within this file you will be able to observe the qualitative and quantative perfomance of two selected models.

All instructions on running the Predictions.ipynb file are included within. The preliminary instruction is that it is highly recommended to run the file in google colaboratory with the GPU accelerator enabled. When using google colaboratory, it must be ensured that the 3 zip files are uploaded in the file section.

Two trained models with different architectures are utilized for this demonstration, the number of models is limited to two as the size of each model is quite large. The models can be downloaded from within the jupyter notebook without any intervention.

After navigating to google colab, click the upload file section and upload Predictions.ipynb.

Link to google colab : https://colab.research.google.com/

For any issues or queries contact : maa089@alumni.bham.ac.uk

# Experimental Analysis on Diabetic Retinopathy Lesion Segmentation

For my final computer science project, I took on the research based problem of Diabetic Retinopathy (DR) detection through lesion segmentation. This project involved applying research in the field of image segmentation to the particularly challenging task of segmenting lesions in fundus images. The project was conducted under the supervision of Dr. Kashif Rajpoot. The project was conducted at the University of Birmingham, School of Computer Science.

My goal for mastering deep learning has always been to bring a positive impact through the field, and this project was inspired by the potential to make a difference in the lives of millions of people starting from my uncle who is afflicted by this condition.  

## Introduction

Medical image segmentation plays a crucial role in the diagnosis and treatment
of many diseases, including diabetic retinopathy (DR). DR is a leading cause
of blindness and involves abnormalities in the blood vessels of the retina. In
this study, we investigate the effects of different U-Net models for DR lesion
segmentation and identify best approaches for more accurate segmentation using limited data. In
this work, we perform experiments on U-Net model architectures, training
approaches, image preprocessing techniques, and loss functions. Our
contributions include identifying binary class map weighting as a solution for
more accurate segmentation of the microaneurysm class. This study provides
insights into improving the accuracy of DR lesion segmentation, which could
ultimately aid in the effective diagnosis and treatment of DR.

In this project, we have conducted experiments to explore different U-Net architectures
and related techniques for DR lesion segmentation. Specifically, we have investigated the
impact of using different pre-trained encoders, attention mechanisms, and multi-task training
approaches on the performance of U-Net-based models for lesion segmentation. We have also
evaluated the effectiveness of various loss functions and preprocessing techniques in enhancing
the segmentation results.

## Technical Approach

For fair and reliable comparisons, we have developed all the models, training pipelines, as well as evaluation pipelines using the PyTorch deep learning framework ourselves. 

The data for this project was sourced from the [retinal-lesions dataset](https://github.com/WeiQijie/retinal-lesions), where the dataset was originally introduced in the paper 'Learn to Segment Retinal Lesions and Beyond' by Qijie Wei and team, presented at the International Conference on Pattern Recognition (ICPR) in 2020 [1]. We would like to express our gratitude to the authors for providing this valuable dataset, which forms the foundation of our research and experimentation.

**Citation:**
Q. Wei, X. Li, W. Yu, et al., "Learn to Segment Retinal Lesions and Beyond," *International Conference on Pattern Recognition (ICPR)*, 2020.

Once more, for reliable results the evaluation pipeline utilized an underlying metric library called [miseval](https://github.com/frankkramer-lab/miseval).

**Citation:**
D. Müller, D. Hartmann, P. Meyer, F. Auer, I. Soto-Rey, F. Kramer, "MISeval: a Metric Library for Medical Image Segmentation Evaluation," *Studies in Health Technology and Informatics*, vol. 294, pp. 33-37, 2022. doi: [10.3233/shti220391](https://doi.org/10.3233/shti220391)

### Demonstration

Link to google colab : https://colab.research.google.com/

For any issues or queries contact : maa089@alumni.bham.ac.uk


## Methodology

![alt text](UNet.png)

The U-Net architecture is a popular choice for medical image segmentation tasks and it serves as the basis for our models. The U-Net consists of an encoder-decoder architecture with skip connections between the subsequent convolutional blocks in the encoder and decoder. The skip connections help in preserving the spatial information and allow for better flow of gradients during training. However, the base U-Net isn't equipped to handle the complex and diverse nature of the DR lesions (with limited data) thus, we have experiemented with different additions to the base U-Net architecture.

### U-Net with Pre-trained Encoders

Using a pre-trained conv-net as the encoder in U-Net architecture is a significant improvement
known as transfer learning. It has been shown to enhance performance, reduce training time, and
computational resources. The pre-trained conv-net enables the U-Net to leverage learned feature
representations, especially when training data is limited.

![alt text](Simple_ResUNet.png)

### U-Net with Attention Mechanism

Attention is a mechanism that was utilized in the Transformer architecture to improve
performance in the task of machine translation by training the model to ‘pay attention’ to words
that offer more context, this mechanism has been adapted to computer vision tasks to train models
to selectively focus on regions of the image that more important to the task. In the context of the
U-Net, the use of attention mechanisms can enhance the features extracted from the encoder
which can improve the performance of the model in image segmentation. This work looks at two
types of attention mechanisms: 

- dual attention
- compact-split-attention

#### Dual Attention

Dual attention is mechanism that enhances a model’s ability to capture global features within
images, it consists of two parts: position attention, and channel attention. 

![Spatial Attention](Spatial_Attention.png)
![Channel Attention](Channel_Attention.png)