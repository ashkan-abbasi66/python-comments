import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape
print(dataset.shape)

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))

convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")
# input/output shape: N*H*W*C; N: number of samples; C: number of input channels
# In `tf.nn.conv2d`, `filter` is a tensor with shape [filter_height, filter_width, in_channels, out_channels]
#       filter shape: h*w*(# of FMs in the previous layer)*(# of FMs); FMs: feature maps
#
# We can also define a conv. layer with `tf.layers.conv2d`. However, in this function `filters` is an integer which is 
# the dimensionality of the output space. Note that `tf.layers` is the official API.

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

print(output.shape)
plt.title("1st feature map")
plt.imshow(output[0, :, :, 0], cmap="gray")
plt.show()
plt.title("2nd feature map")
plt.imshow(output[0, :, :, 1], cmap="gray")
plt.show()
