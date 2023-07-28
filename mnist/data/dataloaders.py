import os
import random
from typing import List

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from mnist_c import MNISTC, _CORRUPTIONS
from mnist_label_shift import MNISTLabelShift

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


def get_dataloaders(
    root_dir: str,
    dataset_names: List[str],
    batch_size: int,
    use_meta_batch=False,
    meta_batch_size=0,
    num_workers=0,
):
    """Get the train and test dataloaders for MNIST, MNIST-C, or MNISTLabelShift."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_datasets = []
    test_datasets = []

    for dataset_name in dataset_names:
        if dataset_name == "mnist":
            train_dataset = datasets.MNIST(
                root_dir, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root_dir, train=False, download=True, transform=transform
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
            transform = transforms.Compose([
                transforms.Resize((28, 28)),  # Resize the image to (28, 28), which is the input size of MNIST
                transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
                transforms.ToTensor(),  # Convert PIL image to a PyTorch tensor
            ])
            train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    if not use_meta_batch:
        print(dataset_names)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
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
            train_dataset, batch_sampler=meta_train_batches, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_sampler=meta_test_batches, num_workers=num_workers
        )

    return train_loader, test_loader
