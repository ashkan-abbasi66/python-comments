# my python comments

* **[graph and ml](https://github.com/ashkan-abbasi66/graph-and-ml)**
* **[pytorch](./pytorch)**
* **[prog](./prog)**
* **[ml](./ml)**

  
<br>Implements\PythonCodes
- [Misc. notes](#miscellaneous-notes)
- [Git](#git)
- [Tensorflow](#tensorflow)
    - [Save model during training](#save-model-during-training)
    - [Keras](#keras)
- [Import Data or a module](#import-data-or-a-module)
- [ToDo](#todo)



# Miscellaneous notes

- **Working with meshgrid**

    ```python
    a = np.array([[1,2,3],
                  [4,5,6]]) # Loosly assume that this is the output of a scalar function over a 2-D space.
    # We need 2-D coordinate arrays to evaluate this function.
    rr = [[0,0,0],[1,1,1]]
    cc=[[0,1,2],[0,1,2]]
    b = a[rr,cc]
    (a == b).all()   # True

    # Instead of constructing two 2-D coordinate arrays to address all possible 
    # coordinates in the function space, I can use 1-D arrays to easily
    # create those 2-D arrays that I need. This is where
    # "meshgrid" is useful.
 
    xx, yy = np.meshgrid(np.linspace(0,2,3), np.linspace(0,1,2))
    bb = a[(yy).astype(int), (xx).astype(int)]
    (bb == a).all()  # True    

    ```

- **Remove elements in a list, given their indices**: You can only use an *integer* (`l.pop(0)`) or *slicing* (`l = l[1::2]`) to remove elements from a list. 

    ```python
    l = ["a","b","c","d","e"]
    del_idx = [1,3]
    l2 = [e for i,e in enumerate(l) if i not in del_idx] # l2 => ['a', 'c', 'e']

    l = ["a","b","c","d","e"]
    keep_idx = [0, 2, 4]
    l2 = [e for i,e in enumerate(l) if i in keep_idx] # l2 => ['a', 'c', 'e']
    ```

- **Copy files in a folder into another folder**:

    ```python
    import os, shutil

    def copytree(src, dst, symlinks=False, ignore=None):
        """
        Found at
        https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
        """
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
    ```

- **Format a number with leading zeros**: `("%d"%int_var).zfill(2)` - `print(f'{int_var:03})'`

- **Python Sequences (Sequence objects)**: 
	- They are ordered.
	- There are 6 types: string (group of characters inside `""` or `''`), list, tuple, byte sequence, byte array, and range object.
	- immutable sequences: 
		- string => `name='abc', name[0]='d' #error`
		- tuple
		- `bytes()`
		```python
        size = 10
        b = bytes(size), print( b )
        print(bytes(“Hey”, ‘utf=8’))
        ```
	- mutable sequences:
		- list
		- `bytearray()`
		`print( bytearray([1, 2, 3, 4]) )`

	- supported operators: +(concatenation), *(repeat), membership(`in`), slicing
	- supported functions: `len()`, `min() & max()`, `index()`, `count()`, 

- [Formatted output tutorial](https://python-course.eu/python-tutorial/formatted-output.php)

- **Python Collections**: unordered and unindexed data structures
	- sets and dictionaries are python collections. They are both mutable (you can add, remove, update). But do note that keys in a dictionary is not mutable. 
	- sets
	```python
    setA = {1,2,3,4,5,4}
    setB = set({10,20,30,20,30})
    setC = set() # {}
    
    ```
    	- sets cannot contain sets, lists, and dictionaries.

- In contrast to `Dict()`, `OrderedDict()` preserves the order in which the keys are inserted. (matters when iterating on it)

- **`os.list` and `os.walk` examples**
    ```python
    import os
    file_list = os.listdir(data_dir)
    for root, dir_names, file_names in os.walk(data_dir):
        print(root, ", ", dir_names, ", ", file_names)
    ```


- **Navigate between code sections in PyCharm**
  - Use `# region NAME` and `# endregion` to divide your code into different folds. Then, use `Ctrl+Alt+.` to get a list of your folds.
  - Use `Alt+Down` and `Alt+Up` to navigate between methods defined in a code.


- Set priority of a process
	```python
	"""
	Set the priority of the process to high in Windows:
	"""
	# Found at
	# https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
	import win32api, win32process, win32con
	
	pid = win32api.GetCurrentProcessId()
	handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
	win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
	```
	If you want to use `REALTIME_PRIORITY_CLASS`, check [here](https://stackoverflow.com/questions/44049685/how-to-set-real-time-priority-with-psutil).



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



## Keras

[Model - MiniGoogleLeNet](./tf-examples/model_tf_keras_minigooglenet_functional.py)

[Model - Allconvnet or ALLCNN](./tf-examples/model_tf_keras_allconvnet_allcnn.py)

Search for Tf-KERAS-*


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



# ToDo

indexing and slicing techniques: `indexing.py`<br>

my [OS Notes](./README_OS_NOTES.MD) - NOT completed
