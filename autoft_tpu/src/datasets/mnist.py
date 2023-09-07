import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from src.datasets.utils import SampledDataset
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST as PyTorchMNIST
from torchvision.datasets import VisionDataset

MNIST_CLASSNAMES = [str(i) for i in range(10)]
MNIST_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                     "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                     "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
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
            self.dataset = SampledDataset(self.dataset, num_samples_per_class=n_examples//self.num_classes)

        self.classnames = self.dataset.classes

    def __str__(self):
        return "MNIST"

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
        self.root_dir = Path(location) / 'MNIST-C'
        self.n_total_mnist = 10000
        if n_examples < 0:
            self.n_examples = self.n_total_mnist
        else:
            self.n_examples = n_examples
        self.num_classes = 10
        self.severity = severity
        self.corruptions = MNIST_CORRUPTIONS
        self.transform = preprocess

        if not self.root_dir.exists():
            os.makedirs(self.root_dir)

        # Load data
        labels_path = self.root_dir / 'labels.npy'
        if not os.path.isfile(labels_path):
            raise ValueError("Labels are missing, try to re-download them.")
        self.labels = np.load(labels_path)
        data, targets = self.load_data()

        # Only use for test
        self.dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(targets).long(), transform=preprocess,
        )

    def load_data(self):
        all_images = []
        all_labels = []
        for corruption in self.corruptions:
            check_exists(self.root_dir, corruption)
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
        labels = np.concatenate(all_labels, axis=0)

        return images, labels

    def __str__(self):
        return "MNISTC"
