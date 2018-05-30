import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

import numpy as np
from sklearn.datasets import load_sample_images

# Load sample images
china = load_sample_image("china.jpg")
#flower = load_sample_image("flower.jpg")
dataset = np.array([china], dtype=np.float32)
nI, height, width, channels = dataset.shape
print(dataset.shape)

# Create 2 filters
filters = np.zeros(shape=(3, 3, channels, 2), dtype=np.float32)
filters[:, :, :, 0] = np.tile([-1,0,1],[3,1])  # vertical line
filters[:, :, :, 1] = np.tile([-1,0,1],[3,1]).T # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

print(output.shape)
plt.title("1st feature map")
plt.imshow(output[0, :, :, 0], cmap="gray")
plt.show()
plt.title("2nd feature map")
plt.imshow(output[0, :, :, 1], cmap="gray")
plt.show()