import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset


class CINIC10(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(CINIC10, self).__init__(root, transform=transform, target_transform=target_transform)

        assert split in ['train', 'valid', 'test']
        self.root = os.path.join(root, split)

        self.data = []
        self.targets = []

        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            for filename in os.listdir(class_path):
                if filename.endswith('.png'):
                    self.data.append(os.path.join(class_path, filename))
                    self.targets.append(class_folder)  # or a dictionary to map class names to IDs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]
