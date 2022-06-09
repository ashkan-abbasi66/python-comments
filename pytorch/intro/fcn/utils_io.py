import torch
import os
import torchvision
import matplotlib.pyplot as plt

# use this to make the datasets smaller
from torch.utils.data import SequentialSampler
import random


def plot_images_in_rows(image_lists, n_rows, save_path = ''):
    n_cols = len(image_lists[0]) # each list has n_cols number of images
    scale = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, 1 * scale))
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].imshow(image_lists[i][j].numpy())  # .numpy()
            axs[i, j].axis('off')
    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path)


def read_voc_images(voc_dir, is_train=True):
    """
    Read all VOC feature and label images.
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')

    mode = torchvision.io.image.ImageReadMode.RGB

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    features, labels = [], []

    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), mode))

    return features, labels


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def voc_colormap2label():
    """
    Build the mapping from RGB to class indices for VOC labels.
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """
    Map any RGB values in VOC labels to their class indices.
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def crop_image(image, crop_rect = (0, 0, 320, 480)):
    # crop_rect: The crop rectangle, as a (left, upper, right, lower)-tuple.
    return torchvision.transforms.functional.crop(image, *crop_rect)


def voc_rand_crop(feature, label, height, width):
    """
    Randomly crop both feature and label images.
    """
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = crop_image(feature, rect)
    label = crop_image(label, rect)

    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """
    A customized dataset to load the VOC dataset.
    """
    @staticmethod
    def normalize_input_image(image):
        transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transform(image.float() / 255)

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.crop_size = crop_size

        features, labels = read_voc_images(voc_dir, is_train=is_train)

        # Only images which are bigger than the crop size are considered.
        self.features = [VOCSegDataset.normalize_input_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)

        self.colormap2label = voc_colormap2label()

        print('read ' + str(len(self.features)) + ' examples')

    # def normalize_image(self, img):
    #     return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def get_voc_dataloaders(voc_dir, batch_size, crop_size):
    """
    batch_size = 64
    crop_size = (320, 480)
    trainloader, testloader = get_voc_dataloaders(batch_size, crop_size)
    """
    trainloader = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=False,
        drop_last=True,
        # sampler=SequentialSampler(random.sample(range(0, 1000), 200))
    )

    testloader = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True,
        # sampler=SequentialSampler(random.sample(range(0, 200), 100))
    )

    return trainloader, testloader