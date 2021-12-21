# my python comments
This repo. contains useful python tips for me!
<br>Implements\PythonCodes

- [object oriented programing](#ood)
- [Image Processing Operations](#dip_ops)
- [Parsing command line arguments](#sysargv)
- [Assert](#exception)
- [Tensorflow](#tf)
    - [Save model during training](#tf-saveModel)
    - [Keras](#keras)
- [Import Data or a module](#loadData)
- [ToDo](#todo)



## Miscellaneous comments

[Quick guide for installing Python, Tensorflow, and PyCharm on Windows](https://medium.com/@ashkan.abbasi/quick-guide-for-installing-python-tensorflow-and-pycharm-on-windows-ed99ddd9598) - After installation, you may need to install some libraries. E.g., `pip install opencv-python pandas pillow jupyter matplotlib sklearn`

[Cool Python Libs](https://github.com/maet3608/cool-python-libs)



<a id="ood"></a>

# object oriented programing
simple examples based on Learn Python in One Day (2nd Edition) - page: 68<br> see `simple-object-oriented-example/main.ipynb`.
<a id="dip_ops"></a>

# Image Processing Operations
- See `guide_image_basics_git.ipynb`: Reading and displaying an image is among the most frequently used operations. However, since there are various ways for doing them in Python, sometimes it is confusing!. In this guide, I will show you the most common ways. I am sure that this will save your time!. Also, there are other tips that might be useful. - matplotlib (Read and Display an image) - PIL (Read and Display an image; Save noisy image;resize;color space) - CV2 (Read and Display an image)
- 

## Imshow and Imread

Using **cv2** <br>

```python
import cv2
import numpy as np

def showImage(title,A):
    cv2.imshow(title, A)
    cv2.waitKey(0)

Y = cv2.imread('barbara.tif',0)/255
sigma=50/255
Yn=Y+sigma*np.random.randn(*Y.shape) # * can be used to unpack a list into its elements.
showImage('original',Y)
showImage('noisy',Yn)
cv2.destroyAllWindows()

```
Using **matplotlib**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Y=mpimg.imread('barbara.tif')
print(Y.dtype) # uint8
Yn=Y+50*np.random.randn(*Y.shape)
scaled_Yn=Yn/255.

fig=plt.figure()
fig.suptitle('noisy')
plt.imshow(scaled_Yn,cmap='gray')
plt.show()
```
Note: Unfortunately, the image is shown like a color image!. Use PIL for reading the image.<br>
You can also define a function to show images using matplotlib:

```python
def showImage(title,A):
    fig = plt.figure()
    fig.suptitle(title)
    plt.imshow(A, origin='image', interpolation="nearest", cmap=plt.cm.gray)
showImage('original',Y/255.)
showImage('noisy',scaled_Yn)
plt.show()
```
Usint **matplotlib** or **PIL (Pillow)** for reading, showing and writing gray-scale images:<br>
```python
# read image using PIL
from PIL import Image
fname = 'barbara.tif'
image = Image.open(fname) #.convert('L') can be used to convert an rgb image to gray-scale

# convert to numpy array
import numpy as np
arr = np.asarray(image)
print(arr.shape) # (512, 512)

# add noise
if arr.max()>1:
    arr=arr/255

arr=arr+20/255*np.random.randn(*arr.shape)

# show the image using matplotlib
import matplotlib.pyplot as plt
plt.imshow(arr, cmap='gray')
plt.show()

# show the image using PIL
noisy_image=Image.fromarray(arr*255.)
noisy_image.show()

# save the image using matplotlib
import matplotlib.image as img
img.imsave("noisy1.tif",arr,format='tif',cmap=plt.get_cmap('gray'))

# save the image using PIL --- the saved image is NOT scaled.
noisy_image=Image.fromarray(np.uint8(arr*255.))
noisy_image.save('noisy2.tif')
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
Folder: `tf-example` <br>

- number of parameters of tensorflow model [here](https://stackoverflow.com/questions/47310132/number-of-cnn-learnable-parameters-python-tensorflow)
- compute receptive field of a network. [here](https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-size)
- tensorflow basics: `tf_basics.ipynb`<br>
- some simple train examples: `tf_train_examples.ipynb`
    - A two layer network using pure numpy
    - using tensorflow's Gradient Descent optimizer
- `tf_train_save_restore.ipynb` contains examples for saving and restoring models.
    - Restore a variable with a different name
- perform convolution for spatial filtering with two simple filters `tensorflow_filtering.py` <br>
- comparisson between gpu and cpu computations `matrixmult_cpu_versus_gpu.py` <br>
- control cpu cores or gpu usage `control_gpu_cpu.py` <br>
- How to design and code nn layers:
> - `building_nnlayers_CAN24.py`
> - `building_nnlayers_DPED_Resnet.py`
<a id="tf-saveModel"></a>
## Save model during training
```python
sess.run(tf.global_variables_initializer())

# create a saver object
saver=tf.train.Saver(max_to_keep=1000)

# get `checkpoint` file if it is available in the directory `checkpoint_dir`
ckpt=tf.train.get_checkpoint_state(checkpoint_dir) # <<<-------------
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
for epoch in range(1,N_epochs):

    # if a model is loaded, we can continue training with the loaded model.
    if os.path.isdir("%d"%epoch):
        continue
    
    # Do computations for each epoch
    
    # After each epoch, create a directory & save the model using saver object.
    os.makedirs("%d"%epoch)
    saver.save(sess,"%d/model.ckpt"%epoch)
    saver.save(sess,checkpoint_dir) # <<<------------- 
    
    # it is a good idea to use "%s/%d/model.ckpt"%(checkpoint_dir,epoch) instead of "%d/model.ckpt"%epoch
    
    # [optional] At the end of each epoch, it is a good idea to evaluate the obtained model.
    
```
1. A good structure is:<br>
-  `checkpoint_dir` or the output directory
    -  After last epoch, save the obtained model here.
-  `checkpoint_dir/epoch number/`
    -  some statistics about each epoch.
    -  obtained validation/test results during training.
    -  save the obtained model in that epoch.
    <br>
2. What do they usually save using `saver.save(sess,path)`? 
<br>
-  a file named `checkpoint` (may contain CheckpointState proto). <br>
-  a file named `model.ckpt.data-00000-of-00001`.
-  a file named `model.ckpt.index`.
-  a file named `model.ckpt.meta`.
At this time, I don't know much about those files!<br>



## Keras<a id = "keras"></a>

[Model - MiniGoogleLeNet](./tf-examples/model_tf_keras_minigooglenet_functional.py)

[Model - Allconvnet or ALLCNN](./tf-examples/model_tf_keras_allconvnet_allcnn.py)

Search for Tf-KERAS-*

<a id="loadData"></a>

# Import Data or a module
`load_dataset.py`

- load image pairs
- load a random subset of patch pairs.<br>

Load a module
```python
module_path='.../module_name.py'
from importlib.machinery import SourceFileLoader
module_name = SourceFileLoader("module_name",module_path).load_module()

```




**Load Numpy data**

`np.load` and `np.save` => a platform-independent way of saving and loading a Numpy arrays



<a id="todo"></a>

# ToDo

indexing and slicing techniques: `indexing.py`<br>

my [OS Notes](./README_OS_NOTES.MD) - NOT completed