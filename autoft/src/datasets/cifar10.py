import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset

CIFAR_CLASSNAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                       "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

class CIFAR10:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.dataset = PyTorchCIFAR10(
            root=location, download=True, train=self.train, transform=preprocess
        )
        if n_examples > -1:
            indices = list(range(n_examples))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        self.classnames = self.dataset.classes

    def __str__(self):
        return "CIFAR10"

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.targets)


class CIFAR10C(Dataset):
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 severity: int = 5):

        assert 1 <= severity <= 5, "Severity level must be between 1 and 5."
        self.train = train
        self.root_dir = Path(location) / 'CIFAR-10-C'
        self.n_examples = n_examples
        self.n_total_cifar = 10000
        self.severity = severity
        self.corruptions = CIFAR10_CORRUPTIONS
        self.transform = preprocess

        if not self.root_dir.exists():
            os.makedirs(self.root_dir)

        # Load data
        labels_path = self.root_dir / 'labels.npy'
        if not os.path.isfile(labels_path):
            raise ValueError("Labels are missing, try to re-download them.")
        self.labels = np.load(labels_path)
        data, targets = self.load_data()

        # Split into train and test
        n_train = int(len(data) * 0.8)
        n_test = len(data) - n_train
        train_data, test_data = random_split(data, [n_train, n_test])
        train_targets, test_targets = random_split(targets, [n_train, n_test])

        if self.train:
            self.dataset = BasicVisionDataset(
                images=train_data, targets=torch.Tensor(train_targets).long(), transform=preprocess,
            )
        else:
            self.dataset = BasicVisionDataset(
                images=test_data, targets=torch.Tensor(test_targets).long(), transform=preprocess,
            )
        if n_examples > -1:
            indices = list(range(n_examples))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def load_data(self):
        x_test_list, y_test_list = [], []
        for corruption in self.corruptions:
            corruption_file_path = self.root_dir / (corruption + '.npy')
            if not corruption_file_path.is_file():
                raise ValueError(f"{corruption} file is missing, try to re-download it.")
            images_all = np.load(corruption_file_path)
            images = images_all[(self.severity - 1) * self.n_total_cifar:self.severity * self.n_total_cifar]
            x_test_list.append(images)
            y_test_list.append(self.labels)

        x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

        return x_test, y_test

    def __str__(self):
        return "CIFAR10C"


class CIFAR101:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        assert train is False, "CIFAR-10.1 only has a test set."
        self.train = train
        data_root = os.path.join(location, "CIFAR-10.1")
        data = np.load(os.path.join(data_root, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_root, 'cifar10.1_v6_labels.npy'), allow_pickle=True)

        self.dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )
        if n_examples > -1:
            indices = list(range(n_examples))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)
        self.classnames = CIFAR_CLASSNAMES

    def __str__(self):
        return "CIFAR101"

class CIFAR102:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        train_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_train.npz'), allow_pickle=True)
        test_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_test.npz'), allow_pickle=True)

        train_data_images = train_data['images']
        train_data_labels = train_data['labels']

        test_data_images = test_data['images']
        test_data_labels = test_data['labels']

        if self.train:
            self.dataset = BasicVisionDataset(
                images=test_data_images, targets=torch.Tensor(test_data_labels).long(),
                transform=preprocess,
            )
        else:
            self.dataset = BasicVisionDataset(
                images=train_data_images, targets=torch.Tensor(train_data_labels).long(),
                transform=preprocess,
            )
        if n_examples > -1:
            indices = list(range(n_examples))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)
        self.classnames = CIFAR_CLASSNAMES

    def __str__(self):
        return "CIFAR102"