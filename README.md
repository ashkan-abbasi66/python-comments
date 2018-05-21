# python-tips
This repo. contains useful python tips for me!

# show image
```python
import cv2
import numpy as np

def showImage(title,A):
    cv2.imshow(title, A)
    cv2.waitKey(0)

Y = cv2.imread('barbara.tif',0)/255
showImage('original',Y)
'''
