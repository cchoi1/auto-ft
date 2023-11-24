import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset

from src.datasets.utils import SampledDataset

CIFAR_CLASSNAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                       "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

class CIFAR10:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.dataset = PyTorchCIFAR10(
            root=location, download=True, train=self.train, transform=preprocess
        )
        self.num_classes = 10
        if n_examples > -1:
            if use_class_balanced:
                self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes)
            else:
                indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)

        self.classnames = CIFAR_CLASSNAMES
    
    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "CIFAR10"

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            if transform.transforms[0] is not convert:
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
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 severity: int = 5):

        assert 1 <= severity <= 5, "Severity level must be between 1 and 5."
        self.train = train
        self.root_dir = Path(location) / 'CIFAR-10-C'
        self.n_total_cifar = 10000
        if n_examples < 0:
            self.n_examples = self.n_total_cifar
        else:
            self.n_examples = n_examples
        self.num_classes = 10
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
        if use_class_balanced:
            data, targets = self.load_balanced_data()
        else:
            print("Using randomly sampled data (not class/corruption balanced)")
            data, targets = self.load_data()
            indices = np.random.choice(len(data), self.n_examples, replace=False)
            data = data[indices]
            targets = targets[indices]

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
    def load_balanced_data(self):
        x_list, y_list = [], []
        num_examples_per_class = self.n_examples // (len(self.corruptions) * self.num_classes)

        for corruption in self.corruptions:
            corruption_file_path = self.root_dir / (corruption + '.npy')
            if not corruption_file_path.is_file():
                raise ValueError(f"{corruption} file is missing, try to re-download it.")

            images_all = np.load(corruption_file_path)
            start_idx = (self.severity - 1) * self.n_total_cifar
            end_idx = self.severity * self.n_total_cifar
            images = images_all[start_idx:end_idx]
            labels_for_corruption = self.labels[start_idx:end_idx]

            smallest_class_count = min(self.n_total_cifar//self.num_classes, num_examples_per_class)
            x_class_balanced, y_class_balanced = [], []
            for cls in range(self.num_classes):
                class_indices = np.where(labels_for_corruption == cls)[0]
                selected_indices = np.random.choice(class_indices, smallest_class_count, replace=False)
                x_class_balanced.append(images[selected_indices])
                y_class_balanced.extend([cls] * smallest_class_count)

            x_list.append(np.concatenate(x_class_balanced, axis=0))
            y_list.append(np.array(y_class_balanced))

        x, y = np.concatenate(x_list), np.concatenate(y_list)
        rand_idx = np.random.permutation(np.arange(len(x)))
        x, y = x[rand_idx], y[rand_idx]

        return x, y

    def load_data(self):
        x_list, y_list = [], []

        for corruption in self.corruptions:
            corruption_file_path = self.root_dir / (corruption + '.npy')
            if not corruption_file_path.is_file():
                raise ValueError(f"{corruption} file is missing, try to re-download it.")

            images_all = np.load(corruption_file_path)
            start_idx = (self.severity - 1) * self.n_total_cifar
            end_idx = self.severity * self.n_total_cifar
            images = images_all[start_idx:end_idx]
            labels_for_corruption = self.labels[start_idx:end_idx]

            x_list.append(images)
            y_list.append(labels_for_corruption)

        x, y = np.concatenate(x_list), np.concatenate(y_list)
        rand_idx = np.random.permutation(np.arange(len(x)))
        x, y = x[rand_idx], y[rand_idx]

        return x, y

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "CIFAR10C"

class CINIC:
    def __init__(
            self,
            preprocess,
            train,
            n_examples,
            use_class_balanced=False,
            location=os.path.expanduser('~/data'),
            batch_size=128,
            num_workers=16):
        """
        Initialize CINIC dataset.

        Args:
        - preprocess: The transformations to be applied on the images.
        - train (bool): If True, loads the training data, else loads the validation/test data.
        - n_examples (int): Number of examples per class. Default is -1, meaning all examples.
        - location (str): Directory with all the images.
        """
        self.train = train
        self.num_classes = 10
        self.split = 'train' if train else 'test'  # Adjust based on your directory names for train/test/valid.
        self.data_root = os.path.join(location, 'CINIC-10', self.split)

        images = []
        labels = []
        for class_idx, cls in enumerate(os.listdir(self.data_root)):
            class_path = os.path.join(self.data_root, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                with Image.open(img_path) as img:
                    images.append(img.copy())
                    labels.append(class_idx)

        combined = list(zip(images, labels))
        random.shuffle(combined)
        images[:], labels[:] = zip(*combined)

        self.dataset = BasicVisionDataset(
            images=images, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )

        if n_examples > -1:
            if use_class_balanced:
                self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes)
            else:
                indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "CINIC"


class CIFAR101:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.num_classes = 10
        data_root = os.path.join(location, "CIFAR-10.1")
        data = np.load(os.path.join(data_root, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_root, 'cifar10.1_v6_labels.npy'), allow_pickle=True)

        self.dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )
        if n_examples > -1:
            if use_class_balanced:
                self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes)
            else:
                indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
        self.classnames = CIFAR_CLASSNAMES

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "CIFAR101"

class CIFAR102:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.num_classes = 10
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
            if use_class_balanced:
                self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes)
            else:
                indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
        self.classnames = CIFAR_CLASSNAMES

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "CIFAR102"