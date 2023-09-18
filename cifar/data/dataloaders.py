import random
from typing import List

from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
from torchvision import datasets, transforms

from cifar_c import CIFAR10C
from cinic import CINIC10
from utils import SampledDataset

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

def get_transform(dataset_name, augment=False):
    augmentations = []
    if augment:
        augmentations = [
            transforms.RandomRotation(15),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    if dataset_name in ["cifar10", "cinic10", "cifar10c"]:
        transform = transforms.Compose(common_transforms + augmentations)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return transform


def get_dataloaders(
    root_dir: str,
    dataset_names: List[str],
    batch_size: int,
    num_samples_per_class: List[int],
    augment=False,
    use_meta_batch=False,
    meta_batch_size=0,
    num_workers=0,
):
    """Get the train and test dataloaders for CIFAR-10, CIFAR-10-C, CINIC."""
    train_datasets = []
    test_datasets = []
    collate_fn = None

    for i, dataset_name in enumerate(dataset_names):
        train_transform = get_transform(dataset_name=dataset_name, augment=augment)
        test_transform = get_transform(dataset_name=dataset_name, augment=False)
        if dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10(
                root_dir, train=True, download=True, transform=train_transform
            )
            test_dataset = datasets.CIFAR10(
                root_dir, train=False, download=True, transform=test_transform
            )
        elif dataset_name == "cifar10c":
            train_dataset = CIFAR10C(
                root_dir, corruption_type="some_type", severity=1, transform=train_transform
            )
            test_dataset = CIFAR10C(
                root_dir, corruption_type="some_type", severity=1, transform=test_transform
            )
        elif dataset_name == "cinic10":
            train_dataset = CINIC10(
                root_dir, split='train', transform=train_transform
            )
            test_dataset = CINIC10(
                root_dir, split='test', transform=test_transform
            )
        if num_samples_per_class[i] > 0:
            train_dataset = SampledDataset(train_dataset, num_samples_per_class[i])
            test_dataset = SampledDataset(test_dataset, num_samples_per_class[i])
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
