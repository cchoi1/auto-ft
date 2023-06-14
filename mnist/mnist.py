#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split

from mnist_c import MNISTC, _CORRUPTIONS
from mnist_label_shift import MNISTLabelShift

def load_mnist(dataset: str, batch_size=128):
    assert dataset in ["mnist", "mnistc", "emnist", "kmnist", "mnist-label-shift"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset == "mnist":
        train_dataset = datasets.MNIST('/iris/u/yoonho/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('/iris/u/yoonho/data', train=False, download=True, transform=transform)
    elif dataset == "mnistc":
        train_dataset = MNISTC('/iris/u/yoonho/data/MNIST-C', corruptions=_CORRUPTIONS, train=True, transform=transform)
        test_dataset = MNISTC('/iris/u/yoonho/data/MNIST-C', corruptions=_CORRUPTIONS, train=False, transform=transform)
    elif dataset == "emnist":
        train_dataset = datasets.EMNIST('/iris/u/yoonho/data/', split='balanced', train=True, transform=transform)
        test_dataset = datasets.EMNIST('/iris/u/yoonho/data/', split='balanced', train=False, transform=transform)
    elif dataset == "kmnist":
        train_dataset = datasets.KMNIST('/iris/u/yoonho/data/', train=True, transform=transform, download=True)
        test_dataset = datasets.KMNIST('/iris/u/yoonho/data/', train=False, transform=transform, download=True)
    elif dataset == "mnist-label-shift":
        train_dataset = MNISTLabelShift('/iris/u/yoonho/data/MNIST', training_size=60000, testing_size=10000,
                                        shift_type=1, parameter=0.5, target_label=1, transform=transform, train=True)
        test_dataset = MNISTLabelShift('/iris/u/yoonho/data/MNIST', training_size=60000, testing_size=10000,
                                        shift_type=1, parameter=0.5, target_label=1, transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
# %%
