import os
from pathlib import Path

import PIL
import numpy as np
import torch
import torchvision
from src.datasets.utils import SampledDataset
from torch.utils.data import Dataset
from torchvision.datasets import MNIST as PyTorchMNIST
from torchvision.datasets import EMNIST as PyTorchEMNIST
from torchvision.datasets import VisionDataset

MNIST_CORRUPTIONS = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "impulse_noise", "motion_blur",
                "rotate", "scale", "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag"]
_TRAIN_IMAGES_FILENAME = 'train_images.npy'
_TRAIN_LABELS_FILENAME = 'train_labels.npy'
_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'

class MNIST:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.dataset = PyTorchMNIST(
            root=location, download=True, train=self.train, transform=preprocess
        )
        self.num_classes = 10
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir="./data")

        self.classnames = self.dataset.classes

    def __str__(self):
        return "MNIST"


class EMNIST:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train = train
        self.dataset = PyTorchEMNIST(
            root=location, split="digits", download=True, train=self.train, transform=preprocess
        )
        self.num_classes = 10
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir="./data")

        self.classnames = self.dataset.classes

    def __str__(self):
        return "EMNIST"


def convert(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)  # Convert grayscale to RGB
        elif x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)  # Convert grayscale to RGB
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


def check_exists(root, corruption):
    """Check if the dataset is present."""
    assert os.path.exists(os.path.join(root, corruption, _TRAIN_IMAGES_FILENAME)) and \
           os.path.exists(os.path.join(root, corruption, _TRAIN_LABELS_FILENAME)) and \
           os.path.exists(os.path.join(root, corruption, _TEST_IMAGES_FILENAME)) and \
           os.path.exists(os.path.join(root, corruption, _TEST_LABELS_FILENAME)), \
        f"Download the dataset first to {root}!"


class MNISTC(Dataset):
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
        self.root = Path(location) / 'MNIST-C'
        self.n_total_mnist = 10000
        if n_examples < 0:
            self.n_examples = self.n_total_mnist
        else:
            self.n_examples = n_examples
        self.num_classes = 10
        self.severity = severity
        self.corruptions = MNIST_CORRUPTIONS
        self.transform = preprocess

        if not self.root.exists():
            os.makedirs(self.root)

        data, targets = self.load_data()

        # Only use for test
        self.dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(targets).long(), transform=preprocess,
        )

    def load_data(self):
        all_images = []
        all_labels = []
        for corruption in self.corruptions:
            check_exists(self.root, corruption)
            if self.train:
                images_file = os.path.join(self.root, corruption, _TRAIN_IMAGES_FILENAME)
                labels_file = os.path.join(self.root, corruption, _TRAIN_LABELS_FILENAME)
            else:
                images_file = os.path.join(self.root, corruption, _TEST_IMAGES_FILENAME)
                labels_file = os.path.join(self.root, corruption, _TEST_LABELS_FILENAME)
            images = np.load(images_file)
            labels = np.load(labels_file)
            all_images.append(images)
            all_labels.append(labels)
        images = np.concatenate(all_images, axis=0)
        print(images.shape)
        labels = np.concatenate(all_labels, axis=0)

        return images, labels

    def __str__(self):
        return "MNISTC"


class RotatedMNIST:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        self.train = train
        self.num_classes = 10

        train_data = np.load(os.path.join(location, "RotatedMNIST", 'rotated_mnist_train.npz'), allow_pickle=True)
        test_data = np.load(os.path.join(location, "RotatedMNIST", 'rotated_mnist_test.npz'), allow_pickle=True)
        train_data_images = train_data['images']
        train_data_labels = train_data['labels']
        test_data_images = test_data['images']
        test_data_labels = test_data['labels']
        print('rmnist', test_data_images.shape)

        if self.train:
            self.dataset = BasicVisionDataset(
                images=train_data_images, targets=torch.Tensor(train_data_labels).long(),
                transform=preprocess,
            )
        else:
            self.dataset = BasicVisionDataset(
                images=test_data_images, targets=torch.Tensor(test_data_labels).long(),
                transform=preprocess,
            )
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir="./data")
        self.classnames = [str(i) for i in range(self.num_classes)]

    def __str__(self):
        return "RotatedMNIST"


class ColoredMNIST:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        self.train = train
        self.num_classes = 10

        train_data = np.load(os.path.join(location, "ColoredMNIST", 'colored_mnist_train.npz'), allow_pickle=True)
        test_data = np.load(os.path.join(location, "ColoredMNIST", 'colored_mnist_test.npz'), allow_pickle=True)
        train_data_images = train_data['images']
        train_data_labels = train_data['labels']
        test_data_images = test_data['images']
        test_data_labels = test_data['labels']
        print('cmnist', test_data_images.shape)

        if self.train:
            self.dataset = BasicVisionDataset(
                images=train_data_images, targets=torch.Tensor(train_data_labels).long(),
                transform=preprocess,
            )
        else:
            self.dataset = BasicVisionDataset(
                images=test_data_images, targets=torch.Tensor(test_data_labels).long(),
                transform=preprocess,
            )
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir="./data")
        self.classnames = [str(i) for i in range(self.num_classes)]

    def __str__(self):
        return "ColoredMNIST"