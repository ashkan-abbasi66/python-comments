import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path_to_image = './data/bird.png'

img_pil = Image.open(path_to_image)
img_np = np.asarray(img_pil, dtype=np.float64)

known = np.zeros_like(img_np)
off = np.array([3, 3])
# known[0::off[1],0::off[0]] = 1

for i in range(img_np.shape[0]):
    for j in range(img_np.shape[1]):
        if i % 3 == 0:
            if j % 3 == 0:
                known[i,j] =1

plt.figure()
plt.imshow(known)
plt.show()
