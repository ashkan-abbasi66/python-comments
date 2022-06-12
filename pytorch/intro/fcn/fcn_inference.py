from utils_metric import iou_coef
from utils_io import *
from utils_model import *
import torch.nn
import time

data_dir = r'E:\POSTDOC\PYTHON_CODES\DATASETS'
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

net.add_module('transpose_conv', torch.nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)


"""
Predictions on some test images
"""

net.to(my_device)

test_features, test_labels = read_voc_images(voc_dir, is_train=False)

checkpoint = torch.load(checkpoint_filename)
net.load_state_dict(checkpoint)

image_lists = []
iou_list = []
n_rows = 3
for i in range(n_rows):
    crop_rect = (0, 0, 320, 480)

    input_image = crop_image(test_features[i], crop_rect)   # [3, 320, 480]
    output = predict(net, input_image, my_device)           # [320, 480]
    pred = label2image(output, my_device)                   # [320, 480, 3]
    label_image = crop_image(test_labels[i], crop_rect)     # [3, 320, 480]

    image_lists.append([input_image.permute(1, 2, 0), pred.cpu(), label_image.permute(1, 2, 0)])


plot_images_in_rows(image_lists, len(image_lists), save_path='')
plot_images_in_rows(image_lists, len(image_lists), save_path='plot_some_test_images')

temp = input('Press Enter to End the Program. ')
