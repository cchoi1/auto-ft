import os
import random
from typing import List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_c import MNISTC, _CORRUPTIONS
from mnist_label_shift import MNISTLabelShift


def meta_batch_sampler(dataset, meta_batch_size, batch_size):
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

def get_dataloaders(
        root_dir: str,
        dataset_names: List[str],
        batch_size: int,
        meta_batch_size: int,
        num_workers: int,
        use_meta_batch = False
):
    """Get the train and test dataloaders for MNIST, MNIST-C, or MNISTLabelShift."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_names == ["mnist"]:
        train_dataset = datasets.MNIST(root_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root_dir, train=False, download=True, transform=transform)
    elif all(dataset in ["mnistc"] + _CORRUPTIONS for dataset in dataset_names):
        data_dir = os.path.join(root_dir, "MNIST-C")
        if "mnistc" in dataset_names:
            corruptions = _CORRUPTIONS
        else:
            corruptions = dataset_names
        train_dataset = MNISTC(data_dir, corruptions=corruptions, train=True, transform=transform)
        test_dataset = MNISTC(data_dir, corruptions=corruptions, train=False, transform=transform)
    elif dataset_names == ["mnist-label-shift"]:
        train_dataset = MNISTLabelShift(root_dir, training_size=60000, testing_size=10000,
                                        shift_type=5, parameter=0.5, target_label=1, transform=transform, train=True,
                                        download=True)
        test_dataset = MNISTLabelShift(root_dir, training_size=60000, testing_size=10000,
                                       shift_type=5, parameter=0.5, target_label=1, transform=transform, train=False,
                                       download=True)

    if not use_meta_batch:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        meta_train_batches = meta_batch_sampler(train_dataset, meta_batch_size, batch_size)
        meta_test_batches = meta_batch_sampler(test_dataset, meta_batch_size, batch_size)

        # Wrap the base dataset with DataLoader using the meta-batches
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=meta_train_batches,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=meta_test_batches,
            num_workers=num_workers
        )

    return train_loader, test_loader