# building_nnlayers_....
# demonstrating how to design & code layers in a deep neural network, esp. CNNs

# Note: this is just a note! you can't run it.

# [CAN24](https://github.com/CQFIO/FastImageProcessing)
# Goal: apptoximating time-consuming image processing operators using CNNs.

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        # shape is a list with 4 elements. the first two elements specifies the filter/kernel size
        #  for example 3x3. the third element shows number of channels (e.g. for gray-scale image
        #  it is 1). the last element specified the number of feature maps (the width of convolutional
        #  layer).
        #  operator // is  floor division. thus, the commands "cx, cy = shape[0]//2, shape[1]//2" gets
        #  center pixel position of the kernel. then for all of the channels (for i in range(shape[2]))
        #  the center pixel of the i-th filter is set to 1.
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def build(input):
# define the network architecture
# basic block: r-dilated convolution (C*r)+adaptive batch normalization(BN)+leaky relu(LR)
# L^0: input, L^1: (C*1+BN+LR), L^2: (C*2+BN+LR), ...,L^7: (C*64+BN+LR), L^8: (C*1+BN+LR), L^9: (C*1+BN+LR)
# all layers perform 3x3 convolution, except the last one which is 1x1.
#
# INPUTS:
#  a placeholder for storing network's input.
#   RGB images: input=tf.placeholder(tf.float32,shape=[None,None,None,3])
#   gray-scale: input=tf.placeholder(tf.float32,shape=[None,None,None,1])
# OUTPUTS:
#   net: network architecture
#
    net=slim.conv2d(input,24,[3,3],
                    rate=1,activation_fn=lrelu,
                    normalizer_fn=nm,weights_initializer=identity_initializer(),
                    scope='g_conv1')
    # conv2d & conv3d may be aliases for convolution2d & convolution3d that are defined in layers.py
    # see 'def convolution' in layers.py
    #
    # rate: dilation rate for dilated convolutions (default is 1)
    # 24 output feature maps (24 filters). kernel size: Cx3x3, C:number of channels
    # normalizer_fn:
    #   normalizer_fn=None.
    #   If a 'normalizer_fn' is  provided (such as 'batch_norm'), it is then applied.
    #   Otherwise, if  'normalizer_fn' is None and a 'biases_initializer' is provided then a 'biases'
    #   variable would be created and added the activations.
    #   some functions:
    #       _fused_batch_norm, batch_norm, bias_add, layer_norm
    #   papers:
    #       "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"- (Submitted on 11 Feb 2015
    #       "Layer Normalization" - (Submitted on 21 Jul 2016)
    # weights_initializer: we can also use 'weights_regularizer'
    #   weights_initializer=initializers.xavier_initializer() (default)
    # Reguralizations:
    #   weights_regularizer=tf.contrib.layers.l1_regularizer
    #   weights_regularizer=slim.l1_regularizer(0.07)
    # Activation functions:
    #   activation_fn=nn.relu
    #   activation_fn=tf.nn.tanh
    #
    net=slim.conv2d(net,24,[3,3],
                    rate=2,activation_fn=lrelu,
                    normalizer_fn=nm,weights_initializer=identity_initializer(),
                    scope='g_conv2')
    net=slim.conv2d(net,24,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,24,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,24,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,24,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,24,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net = slim.conv2d(net, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv_last')
    return net


# we don't know the shape and number of training images.
input=tf.placeholder(tf.float32,shape=[None,None,None,1])
output=tf.placeholder(tf.float32,shape=[None,None,None,1])
network=build(input)
loss=tf.reduce_mean(tf.square(network-output))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
    loss,var_list=[var for var in tf.trainable_variables()])


# Some commands that might be useful for designing a network
# tf.concat([net1,net2,net3,net4,net5,net6],axis=3)
# a = tf.minimum(tf.maximum(net1, 0.0), 1.0) * 255.0
# d1 = tf.exp(-d1 / hp)
# w1 = tf.divide(d1, wsum)
# tf.multiply(w1, a)
# tf.square(a-t1)
# a, b, c, d, e,fff = tf.split(network, 6, axis=3)
