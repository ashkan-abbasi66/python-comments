# my python comments

* **[graph and ml](https://github.com/ashkan-abbasi66/graph-and-ml)**
* **[pytorch](./pytorch)**
* **[prog](./prog)**
* **[ml](./ml)**

  
<br>Implements\PythonCodes
- [Git](#git)
- [Tensorflow](#tf)
    - [Save model during training](#tf-saveModel)
    - [Keras](#keras)
- [Import Data or a module](#loadData)
- [ToDo](#todo)



# Miscellaneous comments

<a id="git"></a>
## Git
- **Convert an existing non-empty directory into a Github repository**
[ref](https://stackoverflow.com/questions/3311774/how-to-convert-existing-non-empty-directory-into-a-git-working-directory-and-pus)
```shell
cd <localdir>
git init
git add .
git commit -m 'message'
git branch -M main
# create a github repository and use its URL in the below command.
git remote add origin <url> # Example: git remote add origin https://github.com/ashkan-abbasi66/vf-PyVisualField.git
git push -u origin main
```


- **Shrink `.git` folder**: `git gc --aggressive --prune` [ref](https://stackoverflow.com/questions/5613345/how-to-shrink-the-git-folder).



<a id="tf"></a>
## Tensorflow
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
### Save model during training
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
