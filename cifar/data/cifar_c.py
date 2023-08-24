import os
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image

class CIFAR10C(VisionDataset):
    def __init__(self, root, corruption_type, severity=1, transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(root, transform=transform, target_transform=target_transform)

        assert corruption_type in ['gaussian_noise', 'motion_blur', 'brightness', ...]  # list all 19 corruptions here
        assert 1 <= severity <= 5

        self.root = os.path.join(root, corruption_type, f"severity{severity}")

        self.data = []
        self.targets = []

        for filename in os.listdir(self.root):
            if filename.endswith('.png'):
                self.data.append(filename)
                class_name = filename.split('_')[1]  # assuming filename format is IDX_classname.png
                self.targets.append(class_name)  # or a dictionary to map class names to IDs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.data[index])
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]
