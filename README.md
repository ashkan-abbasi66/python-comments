# python tips and tricks 
This repo. contains useful python tips for me!

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

