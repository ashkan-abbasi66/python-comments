# python tips and tricks 
This repo. contains useful python tips for me!
<br>Implements\PythonCodes
- [object oriented programing](#ood)
- [show image](#imshow)
- [Parsing command line arguments](#sysargv)
- [Assert](#exception)
- [Tensorflow](#tf)

<a id="ood"></a>
# object oriented programing
simple examples based on Learn Python in One Day (2nd Edition) - page: 68<br> see `simple-object-oriented-example/main.ipynb`.
<a id="imshow"></a>
# show image
```python
import cv2
import numpy as np

def showImage(title,A):
    cv2.imshow(title, A)
    cv2.waitKey(0)

Y = cv2.imread('barbara.tif',0)/255
showImage('original',Y)
```
<a id="sysargv"></a>
# Parsing command line arguments
`sys.argv` is a list which contains the command-line arguments. <br>
`argparse` module makes it easy to write user-friendly command-line interfaces. It automatically generates help and usage messages and raise exceptions.<br>
see `commandline-arguments-examples`:<br>
`example1.py`: `sys.argv`, `string.startswith`, `string.split`<br>
`example2.py`: `argparse.ArgumentParser()`, `add_argument`, `vars(ap.parse_args())`

<a id="exception"></a>
# Assert
`assert condition`
<br>is equivalent to:<br>
```python
if not condition:
    raise AssertionError()
```
<br> Another example:
`assert 2 + 2 == 5, "We've got a problem"`
<br>Note that assert is an statement. So, the following command will not work:
`assert(2 + 2 == 5, "We've got a problem")`
The reason is that bool( (False, "We've got a problem") ) evaluates to True. (a non-empty tuple evaluates to True in a boolean context).

<a id="tf"></a>
# Tensorflow
## spatial filtering
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

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
```
