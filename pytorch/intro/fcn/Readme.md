# Fully Convolutional Network for Semantic Segmentation


## Main files

* `train_fcn.py`: trains an FCN on VOC 2012 dataset for semantic segmentation.

* `inference_fcn.py`: loads the model (`trained_fcn.pt`) and some validation images 
for making predictions.  

* `inference_fcn_one_image.py`:

* `inference_fcn_resnet101.py`: loads pretrained FCN (with resnet-101 backbone).

* `inference_deeplabv3_resnet101.py`


The main utility functions used in the above scripts are saved in the following modules:
- `utils_io.py`
- `utils_model.py`


## How to access layers of a network

Let's load a network architecture:

```
import torchvision
import torch

pretrained_net = torchvision.models.resnet18(pretrained=False)
```

There are multiple ways to access **layers/modules** in a network:

```
for module in pretrained_net.modules():
    if isinstance(module,torch.nn.Linear):
        print(module)  # Linear(in_features=512, out_features=1000, bias=True)
```

If you print this network, you will see that `modules()` lets you access to every layers.

E.g., `len(list(pretrained_net.modules()))` will print 68.

In contrast, `children()`, lets you access layers in a higher level manner. 
E.g., `print(len(list(pretrained_net.children())))` will print 10. 
Have a look at ResNet18 architecture to find out why 10 is the output.  

There is also `named_children()` method by which you can access layers with their names:

```
for module_name, module in pretrained_net.named_children():
    if module_name == 'fc':
        print(module)  # Linear(in_features=512, out_features=1000, bias=True)
```


## Feature Extraction

Given an input image with size 480 x 320 (width x height), what is the output 
size of the last ResNet18's feature map?
```
import torchvision
import torch

pretrained_net = torchvision.models.resnet18(pretrained=False)

wanted_layers = list(pretrained_net.children())[:-2]
net = torch.nn.Sequential(*wanted_layers)

X = torch.rand(size=(1, 3, 320, 480))  # RGB image

print(net(X).shape)  # torch.Size([1, 512, 10, 15])
```
The spatial resolution reduces to 1/32 of the original ones. 


## Pixel-wise prediction

We need to upsample the feature map and predict 21 classes for each pixel. 

### 1. Transform the number of feature maps into the number of classes

ResNet18 transforms 3 (RGB) channels to 512 channels in its last feature map.

There are 21 classes in the PASCAL VOC 2012. So, we can use 1x1 convolution 
to reduce the number of channels.

```
num_classes = 21
net.add_module('final_conv', torch.nn.Conv2d(512, num_classes, kernel_size=1))
```


### 2. Upsample the last feature map

ResNet18 reduced the spatial resolution by 1/32 times. 

How can we obtain the parameters of a transposed convolution 
layer to upsample the last feature map into the original image resolution?  

As mentioned before, the parameters of a convolution (to downsample) and its corresponding 
transposed convolution (to upsample) is the same.

The output shape of a convolution operator can be obtained by

$$o=\lfloor \frac{i + 2\times p - k}{s} \rfloor + 1 $$

Given an input image with size 480x320 (width x height), 
what is the convolution operator that can convert it into 15x10?

**In order to reduce the spatial information `s` times, you can use a convolution operator with
`stride=s`, `kernel size=2×s`, and `padding=s/2`.**

So, we can add a transposed convolution layer with similar arguments to upsample from 15x10 to 480x320.

```
num_classes = 21
net.add_module('final_conv', torch.nn.Conv2d(512, num_classes, kernel_size=1))

st = 32
net.add_module('transpose_conv', torch.nn.ConvTranspose2d(num_classes, num_classes,
                                                          stride=st, 
                                                          kernel_size=2*st, 
                                                          padding=st//2))
                                    
X = torch.rand(size=(1, 3, 320, 480))

print(net(X).shape)  # torch.Size([1, 21, 320, 480])
```


## References and More Reading

- D2L book, [14.9](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) to [14.11](https://d2l.ai/chapter_computer-vision/fcn.html).
- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431-3440. 2015.


## TODO:

- [ ] Dice score
- [ ] write inference on one image for the trained FCN. Compare its result with pretrained models.
- [ ] Full FCN implementation
- [ ] Change dataset (e.g., use CamVid)