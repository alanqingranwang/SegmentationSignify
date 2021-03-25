import os
import numpy as np
from PIL import Image
import numbers
import torch
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import sys
import json
from pprint import pprint
import glob
import random

def pil_loader(path):
    img = Image.open(path)
    return img

class Folder(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples
    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root, train,
        transform=None, loader=pil_loader):
        super().__init__()
        self.root = root
        self.train = train
        # self.transform = transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train' if train else 'test',
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(240, 240))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = (TF.to_tensor(mask)*255)
        return image, mask

    def _gather_files(self):
        samples = []
        img_dir = os.path.join(self.root, 'images/*')
        label_dir = os.path.join(self.root, 'labels/*')
        img_paths = glob.glob(img_dir)
        label_paths = glob.glob(label_dir)
        assert len(img_paths) == len(label_paths), 'Img/label size mismatch'

        # Train/val split
        cutoff = int(len(img_paths) * 0.8)
        if self.train:
            indices = torch.arange(0, cutoff)
        else:
            indices = torch.arange(cutoff, len(img_paths))

        for i in indices:
            samples.append((img_paths[i], label_paths[i]))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clean)
        """
        img_path, label_path = self.samples[index]
            
        img = self.loader(img_path)
        label = self.loader(label_path)
        img, label = self.transform(img, label)
        # print('data', torch.unique(label))
        # if self.transform is not None:
        #     img = self.transform(img)
        #     label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.samples)

def load_denoising(root, train, batch_size,
    transform=None, loader=pil_loader):
    """
    Args:
        root (str): root directory to dataset
        train (bool): train or test
        batch_size (int): e.g. 4
        transform (torchvision.transform): transform to noisy images
    """
    if transform is None:
        # default to center crop the image from 512x512 to 256x256
        transform = transforms.Compose([
        # transforms.Resize(256, interpolation=Image.NEAREST),
        # transforms.RandomCrop(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])
        
    dataset = Folder(root, train, transform=transform, 
        loader=pil_loader)

    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader
