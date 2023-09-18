import os
import torch.nn as nn
from typing import Dict, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms

from data.colored_mnist import get_colored_mnist, get_rotated_mnist
from data.mnist_c import MNISTC, MNIST_CORRUPTIONS
from data.utils import SampledDataset
from data.cifar_c import CIFAR10C, CIFAR10_CORRUPTIONS
from utils import set_seed

cifar10_corruptions = [f"cifar10c-{corruption}" for corruption in CIFAR10_CORRUPTIONS]
mnist_corruptions = [f"mnistc-{corruption}" for corruption in MNIST_CORRUPTIONS]

class GrayscaleToRGB(nn.Module):
    def __init__(self):
        super(GrayscaleToRGB, self).__init__()

    def forward(self, tensor):
        # tensor should be (Batch Size, 1, Height, Width) for grayscale images
        # repeat along the channel dimension to get an RGB image
        rgb_tensor = tensor.repeat(3, 1, 1)
        return rgb_tensor

def get_subset(dataset, num_datapoints):
    set_seed()
    rand_idxs = torch.randperm(len(dataset))[:num_datapoints]
    subset = torch.utils.data.Subset(dataset, rand_idxs)
    return subset

def remove_labels(dataset: Dataset) -> Dataset:
    """Helper function to replace dataset labels with None labels."""
    none_labels = [None for _ in range(len(dataset))]
    new_data = [(data, none_label) for (data, _), none_label in zip(dataset, none_labels)]
    return new_data

def get_transform(dataset_name, output_channels):
    assert output_channels == 1 or output_channels == 3, "num_output_channels must be 1 or 3"

    if dataset_name in ["cifar10", "cifar10c"] + cifar10_corruptions:
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
        transform = transforms.Compose(common_transforms)

    elif dataset_name == "cinic10":
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))
        ]
        transform = transforms.Compose(common_transforms)

    elif dataset_name in ["mnist", "mnist-label-shift", "mnistc", "rotated_mnist"] + mnist_corruptions:
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        if output_channels == 1:
            transform = transforms.Compose(common_transforms)
        else:
            transform = transforms.Compose(common_transforms + [GrayscaleToRGB()])

    elif dataset_name == "emnist":
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1733,), (0.3317,))
        ]
        if output_channels == 1:
            transform = transforms.Compose(common_transforms)
        else:
            transform = transforms.Compose(common_transforms + [GrayscaleToRGB()])

    elif dataset_name == "colored_mnist":
        assert output_channels == 3, "num_output_channels must be 3"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_name == "svhn":
        assert output_channels == 3, "num_output_channels must be 3"
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

    elif dataset_name == "svhn-grayscale":
        assert output_channels == 1, "num_output_channels must be 1"
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=output_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return transform

def get_datasets(
    root_dir: str,
    dataset_names: List[str],
    output_channels: int,
    transform=None
):
    """Get the train and test dataloaders for various datasets."""
    train_datasets = []
    test_datasets = []

    for i, dataset_name in enumerate(dataset_names):
        if transform is None:
            train_transform = get_transform(dataset_name=dataset_name, output_channels=output_channels)
            test_transform = get_transform(dataset_name=dataset_name, output_channels=output_channels)
        else:
            train_transform = transform
            test_transform = transform

        if dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10(root_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root_dir, train=False, download=True, transform=test_transform)
        elif dataset_name in ["cifar10c"] + cifar10_corruptions:
            if dataset_name == "cifar10c":
                corruptions = CIFAR10_CORRUPTIONS
            else:
                start_idx = dataset_name.find("-") + 1
                corruptions = dataset_name[start_idx:]
            train_dataset = CIFAR10C(root_dir, corruptions=corruptions, severity=5, transform=train_transform)
            test_dataset = CIFAR10C(root_dir, corruptions=corruptions, severity=5, transform=test_transform)
        elif dataset_name == "cinic10":
            train_dataset = datasets.ImageFolder(f"{root_dir}/CINIC-10/train", transform=train_transform)
            test_dataset = datasets.ImageFolder(f"{root_dir}/CINIC-10/test", transform=test_transform)
        elif dataset_name == "mnist":
            train_dataset = datasets.MNIST(root_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(root_dir, train=False, download=True, transform=test_transform)
        elif dataset_name == "emnist":
            train_dataset = datasets.EMNIST(root_dir, split="digits", train=True, download=True,
                                            transform=train_transform)
            test_dataset = datasets.EMNIST(root_dir, split="digits", train=False, download=True,
                                           transform=test_transform)
        elif dataset_name in ["mnistc"] + mnist_corruptions:
            data_dir = Path(root_dir) / "MNIST-C"
            if dataset_name == "mnistc":
                corruptions = MNIST_CORRUPTIONS
            else:
                start_idx = dataset_name.find("-") + 1
                corruptions = dataset_name[start_idx:]
            train_dataset = MNISTC(
                data_dir, corruptions=corruptions, train=True, transform=train_transform
            )
            test_dataset = MNISTC(
                data_dir, corruptions=corruptions, train=False, transform=test_transform
            )
        elif dataset_name == "svhn":
            train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
        elif dataset_name == "svhn-grayscale":
            train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
        elif dataset_name == "colored_mnist":
            train_datasets = get_colored_mnist(root_dir, train_transform)
            test_datasets = get_colored_mnist(root_dir, test_transform)
        elif dataset_name == "rotated_mnist":
            train_datasets = get_rotated_mnist(root_dir, train_transform)
            test_datasets = get_rotated_mnist(root_dir, test_transform)
        if dataset_name != "colored_mnist" and dataset_name != "rotated_mnist":
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    return train_dataset, test_dataset


def get_all_datasets(
    root: str,
    pretrain: str,
    id: str,
    id_unlabeled: str,
    ood: str,
    ood_unlabeled: str,
    test: List[str],
    num_ood_for_hp: int,
    num_examples,
    transform=None
):
    all_datasets = dict()
    if pretrain != "clip":
        pretrain_name = [pretrain] if type(pretrain) != list else pretrain
        source_data, _ = get_datasets(root_dir=root, dataset_names=pretrain_name, output_channels=3, transform=transform)
        all_datasets["pretrain"] = get_subset(source_data, num_examples["pretrain"])

    id_name = [id] if type(id) != list else id
    id_data, id_val_data = get_datasets(root_dir=root, dataset_names=id_name, output_channels=3, transform=transform)
    if num_examples["id_unlabeled"] > 0:
        id_unlabeled_name = [id_unlabeled] if type(id_unlabeled) != list else id_unlabeled
        id_unlabeled_data, _ = get_datasets(root_dir=root, dataset_names=id_unlabeled_name, output_channels=3, transform=transform)
        all_datasets["id_unlabeled"] = get_subset(remove_labels(id_unlabeled_data), num_examples["id_unlabeled"])
    all_datasets["id"] = get_subset(id_data, num_examples["id"])
    all_datasets["id_val"] = get_subset(id_val_data, num_examples["id_val"])

    ood_name = [ood] if type(ood) != list else ood
    print('ood_name', ood_name)
    ood_data, _ = get_datasets(root_dir=root, dataset_names=ood_name, output_channels=3, transform=transform)
    if num_examples["ood_unlabeled"] > 0:
        ood_unlabeled_name = [ood_unlabeled] if type(ood_unlabeled) != list else ood_unlabeled
        ood_unlabeled_data, _ = get_datasets(root_dir=root, dataset_names=ood_unlabeled_name, output_channels=3, transform=transform)
        all_datasets["ood_unlabeled"] = get_subset(remove_labels(ood_unlabeled_data), num_examples["ood_unlabeled"])
    all_datasets["ood_subset_for_hp"] = get_subset(ood_data, num_ood_for_hp)
    all_datasets["ood"] = get_subset(ood_data, num_examples["ood"])

    for i, test_dist in enumerate(test):
        test_name = [test_dist] if type(test_dist) != list else test_dist
        _, test_data_i = get_datasets(root_dir=root, dataset_names=test_name, output_channels=3, transform=transform)
        all_datasets[f"test{i+1}"] = get_subset(test_data_i, num_examples["test"])

    return all_datasets
