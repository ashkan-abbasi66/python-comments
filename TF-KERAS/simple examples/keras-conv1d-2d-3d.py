"""
PY38TF25
"""

import tensorflow as tf
import numpy as np

"""
tf.nn.conv1d
    data_format='NWC' - input: A Tensor of rank at least 3
tf.keras.layers.Conv1D
    data_format='channels_last'
As a first layer, set "input_shape" to e.g., (10, 128) or (None, 128)
    => number of 128-dimensional vector sequences is 10 or None.
"""

# input_shape = (1, 6, 20)
# x = tf.random.normal(input_shape)

t = np.array([[1,2,3,4,5,6]], dtype='float32')
t2 = np.expand_dims(t, axis = 2)
print(t.shape)
print(t2.shape)
t2 = np.concatenate((t2,t2,t2), axis = 2)
print(t2.shape)

x = tf.Variable(t2)


print()
print("INPUT: NWC => W-length vectors with C timesteps. Batch size is N")
print(x.shape)
for c in np.arange(x.shape[-1]):
    print(x[0,:,c])

y = tf.keras.layers.Conv1D(filters = 1, kernel_size = 3,
                           input_shape=x.shape[1:])(x)
print()
print("OUTPUT")
print(y.shape) # (1, 4, 1)
for c in np.arange(y.shape[-1]):
    print(y[0,:,c])

# When There are multiple channels, how does each kernel is applied (how does convolution is performed)?
#
# For each channel, an appropriate kernel is provided.
# After convolving each kernel with each channel, the resulting feature maps
# will be summed up element-wise to produce the output feature map.
#
# Here, we have three channels. So, three distinct filters
# each one with size 1*3 is applied to each channel. This leads to
# an output with three channels. Those are summed up to produce one channel output
# since we have one filter.
#
# I don't know the numbers in each filter are distinct or not.
# Let's compute the number of parameters.

input_shape = (1, 6, 3)
inp = tf.keras.Input(shape = input_shape)
out = tf.keras.layers.Conv1D(filters = 2, kernel_size = 3,
                           input_shape= input_shape[1:])(inp)
model = tf.keras.Model(inputs = inp, outputs = out)
print()
model.summary()





