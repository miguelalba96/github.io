---
title       : How to build deep neural networks using Tensorflow low level API (tf.Module)      # Title
author      : Miguel Alba                             # Author Name
date        : 2021-11-28 16:00:00 -0400  # Date
categories  : [Machine Learning, Tensorflow] # Catagories, no more than 2
tags        : [Deep neural networks, Tensorflow, Python, Machine learning]            # Tags, any number
pin         : false                       # Should this post be pinned?
toc         : true                        # Table of Contents?
math        : true                        # Does this post contain math?
comments    : true
image:
    src: /assets/img/python_header_image.jpg # Header image path
    
---

## Introduction

In the era before Google's Tensorflow 2.0 release (mid-2019), working in Tensorflow 1.x required a bit of practice and time to understand the workflow of their core APIs. If you wanted to define, train and evaluate deep neural networks, you were required to create **static graphs** and use `tf.Session`'s for pretty much everything. This made debugging large machine learning extremely complicated and inefficient, decreasing the enthusiasm of new deep learning practitioners to learn TF, which in the end made users migrate to more friendly or pythonic options such as Keras or PyTorch.


Since Tensorflow 2.0 there are two types of APIs used to build and train deep neural networks. The first and most known is `tf.keras` which contains a high amount of tutorials and documentation available across the internet (the most known is the official tensorflow [documentation](https://www.tensorflow.org/api_docs/python/tf/keras)), but what if we don't want to rely on `tf.keras` development to build and train deep neural networks, but make our own layers/models with native Tensorflow code?

This post explains in detail how to define, build and call layers and models using Tensorflow's low level API. 

## Base Neural Networks Class (tf.Module)

As the official [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/Module) says, `tf.Module` is a base class for deep neural networks, quite similar to what we have in PyTorch with `nn.Module`. If we want to build a new layer or model we have to subclass it and initialize the trainable/non-trainable parameters as we want using [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable). 

For this example and as a complement to the recent update of the official documentation (which explains clearly how to build a **Multilayer Perceptron Network - MLP**). We will define a **Convolutional Neural Network CNN**.

### Create layers

1. We define define our initialization methods. In most of the cases we are interested in training *weights* and *biases*

    ```python
    def weights(name, shape, mean=0.0, stddev=0.02):
        # for this case I am using a normal distribution
        var = tf.Variable(tf.random_normal_initializer(mean=mean, stddev=stddev)(shape), name=name)
        return var


    def bias(name, shape, constant=0.0):
        var = tf.Variable(tf.constant_initializer(constant)(shape), name=name)
        return var
    ```

2. We create a class for our layer, in this case a 2D convolutional layer, in contrast to the official TF docs we can create the shapes of the *weights* and *biases* inferring their shapes directly on the first forward pass


    ```python
    class Conv2D(tf.Module):
        """
        Custom Convolutional layer with explicit padding
        """
        def __init__(self, num_filters, kernel_size=3, padding=1, strides=1,
                    act_type='ReLU', name=None):
            """
            :param: num_filters: Number of filters
            :param: kernel_size: kernel size
            :param: padding: explicit padding (0 to not use padding int=>1 to pad input)
            :param: strides: explicit strides
            :param: act_type: activation type
            :param: name: name of the layer
            """
            super(Conv2D, self).__init__(name=name)
            self.filters = num_filters
            self.ks = kernel_size
            self.pad = [[padding] * 2] * 2
            self.s = strides
            self.act_type = act_type

        def build(self, input_shape):
            if not hasattr(self, 'weights'):
                self.weights = weights('weights', 
                                    (self.ks, self.ks, input_shape[-1], self.filters))
            if not hasattr(self, 'bias'):
                self.bias = bias('bias', (self.filters))

        @tf.Module.with_name_scope # keep track of the variables names and their hierarchies
        def __call__(self, input):
            # build weights and biases
            input_shape = input.get_shape()
            self.build(input_shape)

            # explicit zero padding and convolution
            x = tf.pad(input, [[0, 0]] + self.pad + [[0, 0]], constant_values=0)
            x = tf.nn.conv2d(x, self.weights, strides=[self.s]*4, padding='VALID')
            x = tf.nn.bias_add(x, self.bias)

            # activation (in this case only ReLU)
            x = tf.nn.relu(x) if self.act_type == 'ReLU' else x
            return x
    ```

3. We can test if the layer implementation is working correctly by calling it and getting its variables (the weight's tensors):

    ```python
    # create fake image
    x = tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

    conv = Conv2D(num_filters=4, name='test_conv')
    conv(x) # forward pass on the layer

    # then we can print the weights created to verify their values and shapes
    conv.trainable_variables
    ```
    ```

    (<tf.Variable 'test_conv/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>,
    <tf.Variable 'test_conv/weights:0' shape=(3, 3, 3, 4) dtype=float32, numpy=
    array([[[[ 0.00553657,  0.01345694,  0.0160969 , -0.00974667],
            [-0.0174912 , -0.00867653,  0.01673212, -0.01981817],
            [ 0.00546766, -0.00352232,  0.01730617,  0.03613425]],
    
            [[-0.02024776,  0.00949733, -0.01584678, -0.0232379 ],
            [-0.00929332,  0.00113463, -0.00618072, -0.00331176],
            [ 0.01551097, -0.01571138,  0.01730444, -0.00512385]],
    
            [[-0.00026536,  0.02602397,  0.03468624, -0.01000871],
            [-0.00465783, -0.01016334,  0.01106991, -0.02338664],
            [-0.00632419, -0.02121128,  0.02255995, -0.00294858]]],
    
    
            [[[ 0.01078237, -0.01998009, -0.01237592, -0.01769269],
            [ 0.01115903, -0.02444682, -0.02847697, -0.00149765],
            [ 0.01714881,  0.03219299,  0.00256057, -0.01943027]],
    
            [[ 0.03568217, -0.00663988,  0.01306447, -0.02267795],
            [ 0.00643562, -0.02551239, -0.02827096, -0.02343682],
            [-0.00941015, -0.00949762,  0.04840184,  0.00754907]],
    
            [[-0.00122065, -0.0315739 , -0.01874557,  0.00350243],
            [-0.02523333,  0.0312549 , -0.00660984,  0.0077161 ],
            [ 0.01008623, -0.00679884, -0.02994534,  0.00870273]]],
    
    
            [[[ 0.00458873,  0.02161162, -0.00432352,  0.00619686],
            [ 0.00921444, -0.00113679, -0.01196389, -0.0254667 ],
            [ 0.0068634 , -0.00199798,  0.04269401,  0.05141414]],
    
            [[-0.01473054, -0.02008617, -0.02860904,  0.03205349],
            [ 0.01996609, -0.00063833, -0.00017963,  0.00412473],
            [ 0.02097527,  0.03115197,  0.00866693,  0.01411885]],
    
            [[ 0.02391446, -0.01629277,  0.02057827,  0.00664399],
            [ 0.00774666, -0.01262378, -0.0365187 , -0.01844351],
            [-0.00975288, -0.01222847,  0.00185206,  0.01441888]]]],
        dtype=float32)>)

    ```
    Have a look on how the name of the weights maintain an order of **"name_of_layer/name_of_weight"**. 
    
    We can also check the size of the output tensor:
    ```python
    # shapes of the output 
    y.get_shape()
    ```
    ```
    TensorShape([1, 128, 128, 4])
    ```


### Create a model

Once the layers are defined with `tf.Module` we can write a complete model. In this case a CNN. To do this, we first have to define a couple of extra standard operations used in this type of models, e.g. max-pooling, linear/dense layers, etc.

First we start with max pool (quite similar to what can be expected in tensorflow 1.x but without the name scoping):

```python
def max_pool2d(x, size=2, stride=None, padding='VALID', name=None):
    """
    Common max-pooling 2D layer
    """
    # Here we are not explicitly paddding the input as in the Conv2D
    stride = stride or size # if no stride is given use the pool size
    x = tf.nn.max_pool2d(x,
                         ksize=[1, size, size, 1],
                         strides=[1, stride, stride, 1],
                         padding=padding, 
                         name=name)
    return x
```

Then we can refine a bit the `Dense` layer of the TF [documentation](https://www.tensorflow.org/api_docs/python/tf/Module) with the tweaks from the `Conv2D` layer previously defined as follows:

```python
class Dense(tf.Module):
    def __init__(self, num_outputs, act_type=None, name=None):
        super(Dense, self).__init__(name=name)
        self.num_outs = num_outputs
        self.act_type = act_type
  
    def build(self, input_shape):
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', shape=(input_shape[-1], self.num_outs))
        if not hasattr(self, 'bias'):
            self.bias = bias('bias', shape=(self.num_outs))

    @tf.Module.with_name_scope
    def __call__(self, inputs):
        # build weights first call
        self.build(inputs.get_shape())

        # linear operation 
        x = tf.matmul(inputs, self.weights) + self.bias

        # activations 
        if self.act_type == 'ReLU':
            return tf.nn.relu(x)
        elif self.act_type == 'Sigmoid':
            return tf.nn.sigmoid(x)
        else:
            return x
```

Then we write the final model (using again `tf.Module`):

```python
class CNN(tf.Module):
    def __init__(self, name=None):
        super(CNN, self).__init__(name=name)
        with self.name_scope:
            self.conv1 = Conv2D(8, name='C64')
            self.conv2 = Conv2D(16,name='C64')
            self.conv3 = Conv2D(32,name='C64')
            self.conv4 = Conv2D(64,name='C64')
            self.out = Dense(1, name='output', act_type='Sigmoid') 

    @tf.Module.with_name_scope
    def __call__(self, input):
        x = self.conv1(input)
        x = max_pool2d(x)
        x = self.conv2(x)
        x = max_pool2d(x)
        x = self.conv3(x)
        x = max_pool2d(x)
        x = self.conv4(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten op
        output = self.out(x)
        return output
```
In this case the model has 4 conv layers for downsampling and 1 final fully connected layer associated to binary output (note the sigmoid activation). 

We can test this model by simply passing an image and checking the model's output:

```python
# Test the CNN
fake_image = tf.random.uniform((1, 128, 128, 3), dtype=tf.float32) 
net = CNN(name='simple_model')
output = net(x)
print(output) # probability
```
```
tf.Tensor([[0.50065225]], shape=(1, 1), dtype=float32)
```

In order to check the correctness of the name scoping we can always print how some variables may look like when debugging. For example, here I want to check if my last layer follows the appropriate hierarchy (model/layer/name_of_the_weight):
```python
output_variables = [var for var in net.trainable_variables if 'output' in var.name]
print(output_variables)
```
```
[<tf.Variable 'simple_model/output/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>,
 <tf.Variable 'simple_model/output/weights:0' shape=(16384, 1) dtype=float32, numpy=
 array([[ 0.03065354],
        [ 0.0149726 ],
        [-0.02742667],
        ...,
        [-0.01549608],
        [ 0.01788269],
        [ 0.05695942]], dtype=float32)>]
```

We finally have a model!. 

## Further steps

Once the final model is designed we can create a custom training pipeline. This involves using a loop as in PyTorch where we perform training and validation steps.

To write a custom training step we can use `tf.GradientTape` and decorate its function with `@tf.function` to speed up its computation. This tutorial in the official [Tensorflow documentation](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) shows pretty well how the low level training/validation loop logic works. The main difference in this case would be using the model we defined above instead of `tf.keras.Model`.

Additional steps might imply writing a summary to log the training metrics in Tensorboard (using `tf.summary.create_file_writer`) and save checkpoints with the help of a checkpoint manager (`tf.train.Checkpoint` and `tf.train.CheckpointManager`). I will create another blog post showing their usage when writing low level TF pipelines.

For now, this link is sample showing an interesting custom training pipeline that mixes these concepts (summaries and checkpoints) in some models written with `tf.keras.Model` for retina damage detection in OCT scans: [Link](https://github.com/miguelalba96/OCT_project) 

This is a link to the Google Colab to run all the code from this post:
[Link](https://colab.research.google.com/drive/18KOzhewyqBGAw_zNWfKWW1IA85qecn1c?usp=sharing)

