import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
import torchvision.transforms as T
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import math
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def compute_mean_and_std(folder):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    ds = PennFudanDataset(
        folder, transform=A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    return mean, std


def get_data_loaders(
    folder: str,
    train_transforms: list,
    valid_transforms: list,
    batch_size: int = 32, 
    valid_size: float = 0.2, 
    num_workers: int = -1, 
    limit: int = -1, 
):

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None}

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    data_transforms = {
        "train": train_transforms,
        "valid": valid_transforms
    }

    # Create train and validation datasets
    train_data = PennFudanDataset(
        folder, 
        transform=data_transforms["train"]
    )
    
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = PennFudanDataset(
        folder, 
        transform=data_transforms["valid"]
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit
        
    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)  # =

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,  # -
        batch_size=1,  # -
        sampler=valid_sampler,  # -
        num_workers=num_workers,
        shuffle=False
    )

    return data_loaders


class PennFudanDataset(torch.utils.data.Dataset):
    
    mean = torch.Tensor([0.4398, 0.4444, 0.4055])
    std = torch.Tensor([0.2635, 0.2507, 0.2577])
    
    def __init__(self, data_dir, transform):
        
        self.data_dir = data_dir
        self.transforms = transform
        
        # Find images and masks
        self.images = sorted(glob.glob(os.path.join(data_dir, "PNGImages", "*.png")))
        self.masks = sorted(glob.glob(os.path.join(data_dir, "PedMasks", "*.png")))
        
        assert len(self.images) == len(self.masks), f"Corrupted dataset. You have {len(self.images)} images but {len(self.masks)} masks"

    def __getitem__(self, idx):
        
        # load images and masks
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        
        # Convert to numpy and collapse all masks for people to the
        # same (this is semantic segmentation, not instance segmentation)
        mask = np.array(mask)
        idx = mask > 0
        mask[idx] = 1

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        return img, mask

    def __len__(self):
        return len(self.images)
    
    def plot(self, idx, renormalize=True, ax=None, prediction=None):
        
        image, mask = self[idx]
                
        if renormalize:
            
            # Invert the T.Normalize transform
            unnormalize = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean = [ 0., 0., 0. ], std = 1 / PennFudanDataset.std),
                    T.Normalize(mean = -PennFudanDataset.mean, std = [ 1., 1., 1. ])
                ]
            )

            image = (unnormalize(image.numpy().T) * 255).numpy().astype(np.uint8)
        
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
        
        _ = ax[0].imshow(image.T)
        _ = ax[1].imshow(mask * 255, cmap='gray', interpolation='none')
        
        if prediction is not None:
            _ = ax[2].imshow(prediction * 255, cmap='gray', interpolation='none')
        
        [sub.axis("off") for sub in ax.flatten()]
        
        return ax


def plot_results(self, idx, renormalize=True, ax=None, prediction=None):
        
        image, mask = self[idx]
                
        if renormalize:
            
            # Invert the T.Normalize transform
            unnormalize = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean = [ 0., 0., 0. ], std = 1 / PennFudanDataset.std),
                    T.Normalize(mean = -PennFudanDataset.mean, std = [ 1., 1., 1. ])
                ]
            )

            image = (unnormalize(image.numpy().T) * 255).numpy().astype(np.uint8)
        
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
        
        _ = ax[0].imshow(image.T)
        _ = ax[1].imshow(mask * 255, cmap='gray', interpolation='none')
        
        if prediction is not None:
            _ = ax[2].imshow(prediction * 255, cmap='gray', interpolation='none')
        
        [sub.axis("off") for sub in ax.flatten()]
        
        return ax
