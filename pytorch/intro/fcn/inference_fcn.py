from utils_metric import iou_coef
from utils_io import *
from utils_model import *
import torch.nn
import time

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
net.load_state_dict(checkpoint)

test_features, test_labels = read_voc_images(voc_dir, is_train=False)

image_lists = []
iou_list = []
n_rows = 3
for i in range(n_rows):
    crop_rect = (0, 0, 320, 480)

    input_image = crop_image(test_features[i], crop_rect)   # [3, 320, 480]
    # input_image = test_features[i]
    output = predict(net, input_image, my_device)           # [320, 480]
    pred = label2image(output, my_device)                   # [320, 480, 3]
    label_image = crop_image(test_labels[i], crop_rect)     # [3, 320, 480]
    # label_image = test_labels[i]

    image_lists.append([input_image.permute(1, 2, 0), pred.cpu(), label_image.permute(1, 2, 0)])


plot_images_in_rows(image_lists, len(image_lists), save_path='')
plot_images_in_rows(image_lists, len(image_lists), save_path='plot_some_test_images')

temp = input('Press Enter to End the Program. ')
