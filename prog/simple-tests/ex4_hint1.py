import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path_to_image = './data/bird.png'

img_pil = Image.open(path_to_image)
img_np = np.asarray(img_pil, dtype=np.float64)

# img_np[0:10,:] = 0

# print(img_np.min(), img_np.max())


# plt.imshow(img_np/255.0)
# plt.show()
# =======================
# print("PIL image max and min for each channel:", img_pil.getextrema())
#
# img_pil2 = Image.new(img_pil.mode, img_pil.size) # Creates a new image with the given mode and size.
# data = list(img_pil.getdata())
# new_list = []
# for item in data:
#     new_item = tuple(int(ti/255.0) for ti in item)
#     new_list.append(new_item)
# img_pil2.putdata(new_list)
# =======================
# img_pil2.show()

def modify_image(img_np):
    o = img_np.copy()
    o[0:10,:] = 0
    # known = np.ones((img_np.shape[0], img_np.shape[1], img_np.shape[2]))
    known = np.ones_like(o)
    known[0:10,:] = 0
    return o, known

img_np2, known = modify_image(img_np)

plt.figure()
plt.imshow(img_np/255.0)
plt.figure()
plt.imshow(img_np2/255.0)
plt.figure()
plt.imshow(known)
plt.show()



