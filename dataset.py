import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SyntheticDataDataset(Dataset):
    def __init__(self, img_size, num_classes=1000, length=1e6):
        self.img_size = img_size
        self.num_classes = num_classes
        self.length = length

    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx):
        sample = (torch.rand(size=(3, self.img_size, self.img_size), dtype=torch.float32) * 2) - 1
        label = random.randint(0, self.num_classes - 1)
        return sample, label
