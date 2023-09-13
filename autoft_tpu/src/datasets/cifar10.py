import os

import numpy as np
import torch
import torchvision
from PIL import Image
from src.datasets.utils import SampledDataset
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset

CIFAR_CLASSNAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                       "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
SAVE_DIR = "/home/carolinechoi/robust-ft/data/"

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
        self.location = location
        self.dataset = PyTorchCIFAR10(
            root=location, download=True, train=self.train, transform=preprocess
        )
        self.num_classes = 10
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir=SAVE_DIR)

        self.classnames = self.dataset.classes

    def __str__(self):
        return "CIFAR10"

    def __len__(self):
        return len(self.dataset)

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
        self.location = os.path.join(location, "CIFAR-10-C")
        self.n_total_cifar = 10000
        if n_examples < 0:
            self.n_examples = self.n_total_cifar
        else:
            self.n_examples = n_examples
        self.num_classes = 10
        self.severity = severity
        self.corruptions = CIFAR10_CORRUPTIONS
        self.transform = preprocess

        # Load data
        labels_path = os.path.join(self.location, 'labels.npy')
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
    def load_data(self):
        x_list, y_list = [], []
        num_examples_per_class = self.n_examples // (len(self.corruptions) * self.num_classes)

        for corruption in self.corruptions:
            corruption_file_path = os.path.join(self.location, f"{corruption}.npy")
            if not os.path.isfile(corruption_file_path):
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

    def __str__(self):
        return "CIFAR10C"

    def __len__(self):
        return len(self.dataset)

class CINIC:
    def __init__(
            self,
            preprocess,
            train,
            n_examples,
            location=os.path.expanduser('~/data'),
            batch_size=128,
            num_workers=16):
        self.train = train
        self.num_classes = 10
        self.split = 'train' if train else 'test'  # Adjust based on your directory names for train/test/valid.
        self.location = os.path.join(location, 'CINIC-10', self.split)

        images = []
        labels = []
        for class_idx, cls in enumerate(os.listdir(self.location)):
            class_path = os.path.join(self.location, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = np.array(Image.open(img_path))
                if img.shape != (32, 32, 3): # skip the one image that is (32,32)
                    continue
                images.append(img)
                labels.append(class_idx)

        self.dataset = BasicVisionDataset(
            images=np.array(images), targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )

        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir=SAVE_DIR)

    def __str__(self):
        return "CINIC"

    def __len__(self):
        return len(self.dataset)


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
        self.num_classes = 10
        self.location = os.path.join(location, "CIFAR-10.1")
        data = np.load(os.path.join(self.location, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(self.location, 'cifar10.1_v6_labels.npy'), allow_pickle=True)

        self.dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )
        if n_examples > -1:
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir=SAVE_DIR)
        self.classnames = CIFAR_CLASSNAMES

    def __str__(self):
        return "CIFAR101"

    def __len__(self):
        return len(self.dataset)

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
        self.num_classes = 10
        self.location = os.path.join(location, "CIFAR-10.2")
        train_data = np.load(os.path.join(self.location, 'cifar102_train.npz'), allow_pickle=True)
        test_data = np.load(os.path.join(self.location, 'cifar102_test.npz'), allow_pickle=True)

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
            self.dataset = SampledDataset(self.dataset, self.__str__(), num_samples_per_class=n_examples//self.num_classes, save_dir=SAVE_DIR)
        self.classnames = CIFAR_CLASSNAMES

    def __str__(self):
        return "CIFAR102"

    def __len__(self):
        return len(self.dataset)