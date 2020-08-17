# What is this?
This project contains scripts to reproduce experiments from my paper 
[Deep Learning with Gaussian Differential Privacy](https://arxiv.org/abs/1911.11607)
by Zhiqi Bu, Jinshuo Dong, Weijie Su and Qi Long.

# The Problem of Interest
Deep learning models are often trained on datasets that contain sensitive information such as individuals' shopping transactions, personal contacts, and medical records. Many differential privacy definitions arise for the study of trade-off between models' performance and privacy guarantees. We consider a recently proposed privacy definition termed f-differential privacy (https://arxiv.org/abs/1905.02383) for a refined privacy analysis of training neural networks. Using GDP instead of (epsilon,delta)-DP, we can get much better privacy guarantee and alternatively, trade off some privacy for better accuracy.

# Description of Files
You need to install Tensorflow python-package [privacy](https://github.com/tensorflow/privacy) to run the following codes.

## Four datasets:
[mnist_tutorial.py](mnist_tutorial.py): private CNN on MNIST

[adult_tutorial.py](adult_tutorial.py): private NN on Adult data

[imdb_tutorial.py](imdb_tutorial.py): private NN on IMDB reviews (LSTM is too slow but still executable)

[movielens_tutorial.py](movielens_tutorial.py): private NN on MovieLens 1M

## Privacy Accountants
[gdp_accountant.py](gdp_accountant.py) computes the moments accountant (MA), central limit theorem (CLT) and dual relation (Dual) between **\delta,\epsilon,\mu**. This computation does not have any TensorFlow dependencies and is **data-independent**, and thus is extremely fast.

For example, if you run MNIST dataset (60,000 samples) for 15 epochs, batch size 256, noise level 1.3 and \delta 1e-5, then your \epsilon is 

**By Moments Accountant:** 

compute_epsilon(15,1.3,60000,256,1e-5)=1.1912

**By GDP CLT (Uniform subsampling):** 

compute_epsU(15,1.3,60000,256,1e-5)=1.0685

**By GDP CLT (Poisson subsampling):** 

compute_epsP(15,1.3,60000,256,1e-5)=0.8345

## Plots
[mnist_plot.py](mnist_plot.py) together with the saved pickles can easily reproduce the figures in the paper.
