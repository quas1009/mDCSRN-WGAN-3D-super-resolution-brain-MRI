# tensorflow-eager-3D-WGAN-3D-super-resolution-in-brain-MRI
## Introduction
- This project is based on replication of article: 
 [Yuhua Chen et. Efficient and Accurate MRI Super-Resolution using a Generative Adversarial Network and 3D Multi-Level Densely Connected Network](https://arxiv.org/ftp/arxiv/papers/1803/1803.01417.pdf),
where DenseNet-WGAN super-resolution network is designed for 3D brain MRI images. Data includes 3D brain images of 1113 subjects from [Human Connectome](https://db.humanconnectome.org).
- The experiment follows an online manner with Google Cloud Platform/colab. Images are downloaded and deleted continuously from online resource while training.
- This is a Tensorflow version of our project and Pytorch version is uploaded [here](https://github.com/hz2538/E6040-super-resolution-project).
## Prerequisites
- tensorflow
- keras
- scipy
- matplotlib
## Method
### Data Preparation
- **Obtain low resolution images by k-space truncation**
- **Split batches then patches**
![preprocess](https://github.com/quas1009/tensorflow-eager-3D-WGAN-3D-super-resolution-in-brain-MRI/blob/master/figure/preprocess.png
 "Illustration of Data Preparation")
