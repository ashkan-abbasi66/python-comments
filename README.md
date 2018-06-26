# python tips and tricks 
This repo. contains useful python tips for me!
<br>Implements\PythonCodes
- [object oriented programing](#ood)
- [show image](#imshow)
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
Folder: `tf-example` <br>
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
    
    # it is a good idea to use "%d/model.ckpt"%(checkpoint_dir,epoch) instead of "%d/model.ckpt"%epoch
    
    # [optional] At the end of each epoch, it is a good idea to evaluate the obtained model.
    
```
A good structure is:<br>
-  `checkpoint_dir` or the output directory
    -  After last epoch, save the obtained model here.
-  `checkpoint_dir/epoch number/`
    -  some statistics about each epoch.
    -  obtained validation/test results during training.
    -  save the obtained model in that epoch.
What do they usually save using `saver.save(sess,path)`?<br>
-  a file named `checkpoint` \[may contain CheckpointState proto\]
-  `model.ckpt.data-00000-of-00001`
-  `model.ckpt.index`
-  `model.ckpt.meta`

<a id="loadData"></a>
# Import Data
`load_dataset.py`
- load image pairs
- load a random subset of patch pairs.
<a id="todo"></a>
# ToDo
Save model during training
<br>
slicing techniques: `slicing.py`
