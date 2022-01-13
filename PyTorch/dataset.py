import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        self.lr_transform = transforms.Compose(
            [
               transforms.Resize(hr_shape),
               transforms.CenterCrop(hr_shape),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        files = sorted(glob.glob(root + "/*.*"))
        self.files = files#[:int(.25*len(files))]
        print("Length of the dataset is: ", len(self.files))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        # img_hr = self.hr_transform(img)

        return img_lr, img_lr

    def __len__(self):
        return len(self.files)