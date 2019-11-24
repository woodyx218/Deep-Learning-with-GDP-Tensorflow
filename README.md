# What is this?
This project contains scripts to reproduce experiments from the paper 
[Deep Learning with Gaussian Differential Privacy]
by Zhiqi Bu, Jinshuo Dong, Weijie Su and Qi Long.

# The Problem of Interest
Deep learning models are often trained on datasets that contain sensitive information such as
individuals' shopping transactions, personal contacts, and medical records. Many differential privacy definitions arise for the study of trade-off between models' performance and privacy guarantees. We consider a recently proposed privacy definition termed f-differential privacy (https://arxiv.org/abs/1905.02383) for a refined privacy analysis of training neural networks.

# Description of Files
You need to install Tensorflow python-package 'privacy'(https://github.com/tensorflow/privacy) to run the following codes.

## Four datasets:
[mnist_tutorial.py]: private CNN on MNIST

[adult_tutorial.py]: private NN on Adult data

[imdb_tutorial.py]: private NN on IMDB reviews

[movielens_tutorial.py]: private NN on MovieLens 1M

## [AMPfaster.R]

This is an example implementation of SLOPE-AMP converging much faster than other commonly known iterative algorithms including ISTA and FISTA.
