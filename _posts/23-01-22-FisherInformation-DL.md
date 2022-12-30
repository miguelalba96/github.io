---
title       : The relevance of the Fisher Information Matrix in Deep Neural Networks       # Title
author      : Miguel Alba                             # Author Name
date        : 2022-01-22 14:00:00 -0400  # Date
categories  : [Machine Learning, Fisher Information Matrix] # Catagories, no more than 2
tags        : [Deep neural networks, Continual learning, Fisher Information Matrix]            # Tags, any number
pin         : false                       # Should this post be pinned?
toc         : true                        # Table of Contents?
math        : true                        # Does this post contain math?
comments    : true
# image:
    # src: /assets/img/python_header_image.jpg # Header image path
    
---

# Introduction

Deep models have always been considered as black boxes, where provided that sufficient and clean data and a clear problem statement are available, they can produce models with reliable and optimal outcomes. This is taking into account their automatic feature extraction and inductive bias properties.

Explaining the behavior these "black boxes" should not be only focused in finding what set of features in the data resemble the most of the predictions of a model, but also dissecting how the information is store inside of a deep neural network and how to use that information to improve the learning process. 

# Measuring Weights Sensitivity

Suppose we interested in fitting a supervised learning model parameterized by $\theta$ and denoted as the probability distribution $p_{\theta}(y \|x)$, being $x$ and $y$ the observations and labels respectively. We want to find estimates (learn the parameters of the model), which from a frequentist point of view could be done by maximizing the models log-likelihood computing:

$$s(\theta)=\nabla_{\theta} \log{p_{\theta}(y |x)}$$

In statistics, $s(\theta)$ is known as the score function, and describes how sensitive the model is to changes in its parameters $\theta$. The covariance of the score function is the Fisher Information Matrix (FIM) and it is defined as:

$$F(\theta) = \underset{p_{ }{}_{ }{}_{\theta }( y |x)  }{\mathbb{E}}[\nabla_{\theta} \log{p_{\theta}(y |x)} \nabla_{\theta} \log{p_{\theta}(y |x)}^T] $$

The FIM can be seen as a measure of uncertainty over the expected estimate of the score function and it very handy in some interesting applications in deep learning.

The most relevant and well-known use case of the FIM is to serve as a preconditioner matrix in [Natural Gradient Descent (NGD)](https://arxiv.org/abs/1412.1193). Here the FIM captures partial second-order information by substituting the use of the model's Hessian to improve gradient descent. This change can be accomplished thanks to some good [mathematical properties of the FIM](https://agustinus.kristia.de/techblog/2018/03/11/fisher-information/) and its relative "convenience" in terms of computation and implementation, which we will discuss [in this post](#how-to-estimate-the-fim-for-a-deep-neural-network).


# How to estimate the FIM for a deep neural network

<!-- Deep neural networks are known for being universal function approximations
In machine learning specially deep neural networks it represents how sensitive is the model to changes on its parameters. This means higher the fisher information, more important is the parameter for a model    -->

# Implications in Continual Learning


# FIM as over-parameterization statistic


# What else can you do with the FIM?

# References

* Ly, Alexander, et al. “A tutorial on Fisher information.” Journal of Mathematical Psychology 80 (2017): 40-55.

* Kunstner, Frederik, et al. Limitations of the Empirical Fisher Approximation for Natural Gradient Descent (2019)


