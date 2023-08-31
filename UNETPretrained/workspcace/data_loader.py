import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F

class CustomDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.input_files = sorted(os.listdir(input_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        input_image = read_image(input_path)
        mask_image = read_image(mask_path)

        if self.transform:
            input_image = self.transform(input_image)
            mask_image = self.transform(mask_image)

        return input_image, mask_image
