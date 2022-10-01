"""
Resource:
https://colab.research.google.com/github/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb#scrollTo=2AB5w01teQ4-
"""

from torchvision import models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch
from utils_io import label2image, VOC_CLASSES
import os
import numpy as np

"""
Load the pretrained model
"""
os.environ['TORCH_HOME'] = '../../../../trained_models/' #setting the environment variable
# https://stackoverflow.com/questions/52628270/is-there-any-way-i-can-download-the-pre-trained-models-available-in-pytorch-to-a

fcn = models.segmentation.fcn_resnet101(pretrained=True)
fcn.eval()

"""
Read an image and prepare it
"""
data_dir = r'../../../../../PYTHON_CODES/DATASETS'
voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
print("\nINFO: dataset directory:\n", voc_dir)

img_dir = os.path.join(voc_dir, 'JPEGImages')
img_fname = '2007_000783.jpg'

img = Image.open(os.path.join(img_dir, img_fname))
plt.imsave('input_image.jpg', np.array(img))

# Apply the transformations needed

transpose = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp = transpose(img).unsqueeze(0)


"""
Forward pass
"""
# NOTE:
#   Usually, the output of a model is a torch.Tensor. But, the output of
#   a torchvision model is an OrderedDict.
#   In the inference mode, the only key is 'out'.
out = fcn(inp)['out']
print (out.shape)  # torch.Size([1, 21, 224, 224])


"""
save the output as an image
"""
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)     # (224, 224)

rgb = label2image(torch.argmax(out.squeeze(), dim=0), 'cpu')
rgb = np.array(rgb).astype('uint8')
plt.imsave('output_image_fcn_resnet101.jpg', np.array(rgb))

class_numbers  = list(np.unique(om))
print([VOC_CLASSES[cn] for cn in class_numbers])