import os

import torch
from torchvision import datasets, transforms

from mnist_c import MNISTC, _CORRUPTIONS
from mnist_label_shift import MNISTLabelShift


def load_dataset(root_dir: str, dataset: str, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset == "mnist":
        train_dataset = datasets.MNIST(root_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root_dir, train=False, download=True, transform=transform)
    elif dataset in ["mnistc"] + _CORRUPTIONS:
        data_dir = os.path.join(root_dir, "MNIST-C")
        if dataset == "mnistc":
            corruptions = _CORRUPTIONS
        else:
            corruptions = [dataset]
        train_dataset = MNISTC(data_dir, corruptions=corruptions, train=True, transform=transform)
        test_dataset = MNISTC(data_dir, corruptions=corruptions, train=False, transform=transform)
    elif dataset == "emnist":
        train_dataset = datasets.EMNIST(root_dir, split='balanced', train=True, transform=transform)
        test_dataset = datasets.EMNIST(root_dir, split='balanced', train=False, transform=transform)
    elif dataset == "mnist-label-shift":
        train_dataset = MNISTLabelShift(root_dir, training_size=60000, testing_size=10000,
                                        shift_type=1, parameter=0.5, target_label=1, transform=transform, train=True,
                                        download=True)
        test_dataset = MNISTLabelShift(root_dir, training_size=60000, testing_size=10000,
                                       shift_type=1, parameter=0.5, target_label=1, transform=transform, train=False,
                                       download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
