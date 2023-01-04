"""
Albumentations is a Python library for fast and flexible image augmentations.

https://albumentations.ai/

"""

import torch
from torchvision import datasets, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2  # converts a NumPy array to a PyTorch tensor

import numpy as np


"""
Creating a Dataset object using Torchvision's dataset
"""
print("Create a Dataset object and apply transforms ...")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST(root = '../../DATA/MNIST', download=True, train=True,
                          transform=transform)

image, label = trainset[0]
print(type(image))    # torch.Tensor
print(image.shape)    # CHW  - [1, 28, 28]
print(len(trainset))


"""
Creating a custom Dataset object and Using Albumentations
"""
print("\nCreate a Dataset object and apply transforms using Albumentations ...")

train_transform = A.Compose([A.Normalize((0.5,), (0.5,)),
                             ToTensorV2()
                             ])

class myMnistDataset(torch.utils.data.Dataset):
    def __init__(self,
                 #images_filepaths,
                 transform=None):
        # self.images_filepaths = images_filepaths
        self.trainset = datasets.MNIST(root='../../DATA/MNIST', download=True, train=True)
        self.transform = transform

    def __len__(self):
        # num_samples = len(self.images_filepaths)
        num_samples = len(self.trainset)
        return num_samples

    def __getitem__(self, idx):
        # image_filepath = self.images_filepaths[idx]
        # image = cv2.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, label = self.trainset[idx]  # returns a PIL image
        image = np.array(image)
        image = np.expand_dims(image, 2)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


trainset2 = myMnistDataset(transform=train_transform)

image, label = trainset2[0]
print(type(image))    # torch.Tensor
print(image.shape)    # CHW  - [1, 28, 28]


"""
Using DataLoader
"""
print("\nReading mini-batches ... ")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)

images, labels = next(dataiter)
print(type(images))  # torch.Tensor
print(images.shape)  # NCHW


"""
Using DataLoader when Dataset object uses Albumentations
"""
print("\nReading mini-batches processed with Albumentations ...")
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=64, shuffle=True)
dataiter2 = iter(trainloader2)

images2, labels2 = next(dataiter2)
print(type(images2))  # torch.Tensor
print(images2.shape)  # NCHW