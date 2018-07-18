# python tips and tricks 
This repo. contains useful python tips for me!
<br>Implements\PythonCodes
- [object oriented programing](#ood)
- [imread & imshow](#imshow)
- [Parsing command line arguments](#sysargv)
- [Assert](#exception)
- [Tensorflow](#tf)
    -  [Save model during training](#tf-saveModel)
- [Import Data](#loadData)
- [ToDo](#todo)

<a id="ood"></a>
# object oriented programing
simple examples based on Learn Python in One Day (2nd Edition) - page: 68<br> see `simple-object-oriented-example/main.ipynb`.
<a id="imshow"></a>
# imread & imshow
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
Note: cv2.imread is better based on my experience.
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

<a id="loadData"></a>
# Import Data
`load_dataset.py`
- load image pairs
- load a random subset of patch pairs.
<a id="todo"></a>
# ToDo
indexing and slicing techniques: `indexing.py`<br>

