# Python



**[Practical Python Course](./dabeaz/)**

[Udemy-bootcamp](./udemy-bootcamp)


Textbooks:

[Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

[Python Crash Course, Second Edition](https://ehmatthes.github.io/pcc_2e/regular_index/)


# Misc. Notes
- In contrast to `Dict()`, `OrderedDict()` preserves the order in which the keys are inserted. (matters when iterating on it)

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

- [Quick guide for installing Python, Tensorflow, and PyCharm on Windows](https://medium.com/@ashkan.abbasi/quick-guide-for-installing-python-tensorflow-and-pycharm-on-windows-ed99ddd9598) - After installation, you may need to install some libraries. E.g., `pip install opencv-python pandas pillow jupyter matplotlib sklearn`

- [object oriented programing](#ood)
- [Parsing command line arguments](#sysargv)
- [Assert](#exception)
- [Image Processing Operations](#imageProcessing)

<a id="ood"></a>

# object oriented programing
simple examples based on Learn Python in One Day (2nd Edition) - page: 68<br> see `simple-object-oriented-example/main.ipynb`.
<a id="dip_ops"></a>


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

<a id="imageProcessing"></a>
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