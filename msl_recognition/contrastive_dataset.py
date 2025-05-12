"""
SimCLR-style two-view dataset for contrastive learning.
"""
import torch
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, RandomApply

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.augment = transforms.Compose([
            RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.augment(img), self.augment(img)
