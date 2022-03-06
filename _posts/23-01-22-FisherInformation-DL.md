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

Suppose we interested in fitting a supervised learning model parameterized by $\theta$ and denoted as the probability distribution $p_{\theta}(y \|x)$, being $x$ and $y$ the observations and labels respectively. We want to find estimates (learn the parameters of the model), which from a frequentist point of view could be done by maximizing the models log-likelihood computing:

$$s(\theta)=\nabla_{\theta} \log{p_{\theta}(y |x)}$$

In statistics, $s(\theta)$ is known as the score function, and describes how sensitive the model is to changes in its parameters $\theta$.

<!-- Why is this important? The Fisher information plays a role in determining .. -->


<!-- ## A more precise definition -->

When developing a particular estimate for a parameter $/theta$, this has to be inferred from the data 

# How the fisher information relates to deep neural networks 

<!-- Deep neural networks are known for being universal function approximations
In machine learning specially deep neural networks it represents how sensitive is the model to changes on its parameters. This means higher the fisher information, more important is the parameter for a model    -->

# Implications in continual learning


# The Fisher information matrix as over-parameterization statistic


# What else can you do with the FIM?

# References

* Ly, Alexander, et al. “A tutorial on Fisher information.” Journal of Mathematical Psychology 80 (2017): 40-55.

