# NIPS2024 Submission #8992: Temporal Progressive Learning for Spiking Neural Networks
This code is the PyTorch implementation code for [Temporal Progressive Learning for Spiking Neural Networks].

## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

 # Dataset Preparation for CIFAR10/100 and ImageNet
To proceed, please download the CIFAR10/100 and ImageNet datasets on your own.

# Dataset Preparation for DVS-CIFAR10
For CIFAR10-DVS dataset, please refer the Google Drive link below:
Training set: [1](https://drive.google.com/file/d/1pzYnhoUvtcQtxk_Qmy4d2VrhWhy5R-t9/view?usp=sharing)
Test set: [2](https://drive.google.com/file/d/1q1k6JJgVH3ZkHWMg2zPtrZak9jRP6ggG/view?usp=sharing)

# Cofe for TPL
We provide the code for CIFAR10/100/DVS-CIFAR10.

## Train
execute: \
  `bash main.sh`

## Eval
execute: \
  `bash eval.sh`

# Code Reference
Our code is developed based on the code from [Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu. Temporal efficient training of spiking neural network via gradient re-weighting. arXiv preprint arXiv:2202.11946, 2022.].

