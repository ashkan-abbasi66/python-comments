"""
Resource:
https://colab.research.google.com/github/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb#scrollTo=2AB5w01teQ4-
"""

from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

# fcn = models.segmentation.fcn_resnet101(pretrained=True)
# fcn.eval()

# Read an image
data_dir = r'../../../../../PYTHON_CODES/DATASETS'
voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
print("\nINFO: dataset directory:\n", voc_dir)

img_dir = os.path.join(voc_dir, 'JPEGImages')
img_fname = '2007_000783.jpg'

img = Image.open(os.path.join(img_dir, img_fname))
plt.imsave('input_image.jpg', np.array(img))

# Apply the transformations needed
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)