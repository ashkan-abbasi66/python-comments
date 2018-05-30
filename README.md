# python tips and tricks 
This repo. contains useful python tips for me!
<br>Implements\PythonCodes

# object oriented programing
simple examples based on Learn Python in One Day (2nd Edition) - page: 68<br> see `simple-object-oriented-example/main.ipynb`.
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
# Parsing command line arguments
`sys.argv` is a list which contains the command-line arguments. <br>
`argparse` module makes it easy to write user-friendly command-line interfaces. It automatically generates help and usage messages and raise exceptions.<br>
see `commandline-arguments-examples`:<br>
`example1.py`: `sys.argv`, `string.startswith`, `string.split`<br>
`example2.py`: `argparse.ArgumentParser()`, `add_argument`, `vars(ap.parse_args())`
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
