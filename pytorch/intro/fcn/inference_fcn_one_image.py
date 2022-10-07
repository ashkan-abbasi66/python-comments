from utils_metric import iou_coef
from utils_io import *
from utils_model import *
import torch.nn
import time
from PIL import Image
import numpy as np

data_dir = r'../../../../../PYTHON_CODES/DATASETS'
voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
print(voc_dir)

checkpoint_filename = 'trained_fcn.pt'

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))


"""
Build model
"""
pretrained_net = torchvision.models.resnet18(pretrained=True)

wanted_layers = list(pretrained_net.children())[:-2]
print(len(wanted_layers))
net = torch.nn.Sequential(*wanted_layers)

num_classes = 21
net.add_module('final_conv', torch.nn.Conv2d(512, num_classes, kernel_size=1))

st = 32
net.add_module('transpose_conv', torch.nn.ConvTranspose2d(num_classes, num_classes,
                                                          stride=st,
                                                          kernel_size=2*st,
                                                          padding=st//2))
W = bilinear_kernel(num_classes, num_classes, 2*st)
net.transpose_conv.weight.data.copy_(W)


"""
Predictions on some test images
"""

net.to(my_device)

checkpoint = torch.load(checkpoint_filename,
                        map_location=torch.device(my_device))
# RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
# https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device

net.load_state_dict(checkpoint)

# test_features, test_labels = read_voc_images(voc_dir, is_train=False)

data_dir = r'../../../../../PYTHON_CODES/DATASETS'
voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
print("\nINFO: dataset directory:\n", voc_dir)

img_dir = os.path.join(voc_dir, 'JPEGImages')
# img_fname = '2007_000783.jpg'
img_fname = '2007_000033.jpg'

# plt.imsave('input_image.jpg', np.array(rgb))

# img = Image.open(os.path.join(img_dir, img_fname))

mode = torchvision.io.image.ImageReadMode.RGB
img_tensor = torchvision.io.read_image(os.path.join(img_dir, img_fname))

plt.imsave('input_image.jpg', img_tensor.permute((1,2,0)).numpy())

# inp = VOCSegDataset.normalize_input_image(img_tensor)
# ===> no need to do any preprocessing step here because
# we will call do the required preprocessing in the "predict" function later


"""
Forward pass
"""

output = predict(net, img_tensor, my_device)         # [320, 480]


"""
save the output as an image
"""
rgb = label2image(output, 'cpu')
rgb = np.array(rgb).astype('uint8')
plt.imsave('output_image_fcn.jpg', np.array(rgb))

class_numbers  = list(np.unique(output))
print([VOC_CLASSES[cn] for cn in class_numbers])


temp = input('Press Enter to End the Program. ')
