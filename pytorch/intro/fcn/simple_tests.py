import torchvision
import torch

pretrained_net = torchvision.models.resnet18(pretrained=False)

wanted_layers = list(pretrained_net.children())[:-2]
net = torch.nn.Sequential(*wanted_layers)

num_classes = 21
net.add_module('final_conv', torch.nn.Conv2d(512, num_classes, kernel_size=1))

net.add_module('transpose_conv', torch.nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

X = torch.rand(size=(1, 3, 320, 480))

print(net(X).shape)

