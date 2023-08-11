import os
import random
import torch.nn as nn
from typing import List

from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
from torchvision import datasets, transforms

from .colored_mnist import get_colored_mnist, get_rotated_mnist
from .mnist_c import MNISTC, _CORRUPTIONS
from .mnist_label_shift import MNISTLabelShift
from .utils import SampledDataset


def meta_batch_sampler(dataset, meta_batch_size, batch_size):
    assert meta_batch_size > 0
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)

    meta_batches = []
    current_batch = []
    for idx in indices:
        current_batch.append(idx)
        if len(current_batch) == meta_batch_size * batch_size:
            meta_batches.append(current_batch)
            current_batch = []

    if current_batch:
        meta_batches.append(current_batch)

    return meta_batches


class GrayscaleToRGB(nn.Module):
    def __init__(self):
        super(GrayscaleToRGB, self).__init__()

    def forward(self, tensor):
        # tensor should be (Batch Size, 1, Height, Width) for grayscale images
        # repeat along the channel dimension to get an RGB image
        rgb_tensor = tensor.repeat(3, 1, 1)
        return rgb_tensor


def get_transform(dataset_name, output_channels):
    assert output_channels == 1 or output_channels == 3, "num_output_channels must be 1 or 3"
    if dataset_name in ["mnist", "mnist-label-shift", "mnistc", "rotated_mnist"] + _CORRUPTIONS:
        if output_channels == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                GrayscaleToRGB()]
            )
    elif dataset_name == "emnist":
        if output_channels == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,))]
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,)),
                GrayscaleToRGB()]
            )
    elif dataset_name == "colored_mnist":
        assert output_channels == 3, "num_output_channels must be 3"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif dataset_name == "svhn":
        assert output_channels == 3, "num_output_channels must be 3"
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize the image to (28, 28), which is the input size of MNIST
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
    elif dataset_name == "svhn-grayscale":
        assert output_channels == 1, "num_output_channels must be 1"
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize the image to (28, 28), which is the input size of MNIST
            transforms.Grayscale(num_output_channels=output_channels),  # Convert the image to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

    return transform


def get_dataloaders(
    root_dir: str,
    dataset_names: List[str],
    output_channels: int,
    batch_size: int,
    num_samples_per_class=-1,
    use_meta_batch=False,
    meta_batch_size=0,
    num_workers=0,
):
    """Get the train and test dataloaders for MNIST, MNIST-C, or MNISTLabelShift."""
    train_datasets = []
    test_datasets = []
    collate_fn = None

    for dataset_name in dataset_names:
        transform = get_transform(dataset_name=dataset_name, output_channels=output_channels)
        if dataset_name == "mnist":
            train_dataset = datasets.MNIST(
                root_dir, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root_dir, train=False, download=True, transform=transform
            )
        elif dataset_name == "emnist":
            train_dataset = datasets.EMNIST(
                root_dir, split="digits", train=True, download=True, transform=transform
            )
            test_dataset = datasets.EMNIST(
                root_dir, split="digits", train=False, download=True, transform=transform
            )
        elif dataset_name in ["mnistc"] + _CORRUPTIONS:
            data_dir = os.path.join(root_dir, "MNIST-C")
            if "mnistc" in dataset_name:
                corruptions = _CORRUPTIONS
            else:
                corruptions = dataset_name
            train_dataset = MNISTC(
                data_dir, corruptions=corruptions, train=True, transform=transform
            )
            test_dataset = MNISTC(
                data_dir, corruptions=corruptions, train=False, transform=transform
            )
        elif dataset_name == "mnist-label-shift":
            train_dataset = MNISTLabelShift(
                root_dir,
                training_size=60000,
                testing_size=10000,
                shift_type=5,
                parameter=0.5,
                target_label=1,
                transform=transform,
                train=True,
                download=True,
            )
            test_dataset = MNISTLabelShift(
                root_dir,
                training_size=60000,
                testing_size=10000,
                shift_type=5,
                parameter=0.5,
                target_label=1,
                transform=transform,
                train=False,
                download=True,
            )
        elif dataset_name == "svhn":
            train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        elif dataset_name == "svhn-grayscale":
            train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        elif dataset_name == "colored_mnist":
            train_datasets = get_colored_mnist(root_dir, transform)
            test_datasets = get_colored_mnist(root_dir, transform)
        elif dataset_name == "rotated_mnist":
            train_datasets = get_rotated_mnist(root_dir, transform)
            test_datasets = get_rotated_mnist(root_dir, transform)
        if num_samples_per_class > 0:
            train_dataset = SampledDataset(train_dataset, num_samples_per_class)
            test_dataset = SampledDataset(test_dataset, num_samples_per_class)
        if dataset_name != "colored_mnist" and dataset_name != "rotated_mnist":
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    if not use_meta_batch:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )
    else:
        meta_train_batches = meta_batch_sampler(
            train_dataset, meta_batch_size, batch_size
        )
        meta_test_batches = meta_batch_sampler(
            test_dataset, meta_batch_size, batch_size
        )
        # Wrap the base dataset with DataLoader using the meta-batches
        train_loader = DataLoader(
            train_dataset, batch_sampler=meta_train_batches, num_workers=num_workers, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_sampler=meta_test_batches, num_workers=num_workers, collate_fn=collate_fn
        )

    return train_loader, test_loader
