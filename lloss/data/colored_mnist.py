import colorsys
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from utils import set_seed

NUM_COLORS = 100
TRAIN_N_SAMPLES = 50000
TEST_N_SAMPLES = 10000

def color_mnist_image(img, color, subsample=False):
    img = np.array(img).astype(np.float32)
    if subsample:
        img = img[::2, ::2]
    np_digit_color = np.array(color).astype(np.float32)
    np_digit = np.einsum("hw,c->chw", img, np_digit_color)
    return np_digit

def get_single_color_mnist(dataset, color, domain):
    colored_xs, ys, domains = [], [], []
    set_seed(0)
    for idx in range(len(dataset)):
        _x, _y = dataset[idx]
        _x = _x.squeeze(0)
        colored_x = color_mnist_image(_x, color=color)
        # Ensure the image is 4D (batch_size, channels, height, width)
        if len(colored_x.shape) == 3:
            colored_x = np.expand_dims(colored_x, axis=0)
        colored_xs.append(colored_x)
        ys.append(_y)
        domains.append(domain)

    return np.concatenate(colored_xs), np.array(ys)

def get_colored_mnist(root_dir, transform, ratios=np.ones(NUM_COLORS)):
    set_seed(0)
    RGB_tuples = [colorsys.hsv_to_rgb(i / NUM_COLORS, 1.0, 1.0) for i in range(NUM_COLORS)]
    random.shuffle(RGB_tuples)

    if len(ratios) == 1:
        train_ranges = [np.arange(TRAIN_N_SAMPLES)]
        test_ranges = [np.arange(TEST_N_SAMPLES)]
    else:
        ratios = np.array(ratios) / np.sum(ratios)
        cutoffs = np.concatenate([np.array([0]), np.cumsum(ratios)])
        cutoffs[-1] = 1.0
        cutoff_pairs = list(zip(cutoffs[:-1], cutoffs[1:]))
        train_ranges = [
            np.arange(int(TRAIN_N_SAMPLES * c_l), int(TRAIN_N_SAMPLES * c_r))
            for c_l, c_r in cutoff_pairs
        ]
        assert sum(len(r) for r in train_ranges) == TRAIN_N_SAMPLES
        test_ranges = [
            np.arange(int(TEST_N_SAMPLES * c_l), int(TEST_N_SAMPLES * c_r))
            for c_l, c_r in cutoff_pairs
        ]
        assert sum(len(r) for r in test_ranges) == TEST_N_SAMPLES

    mnist_train = torchvision.datasets.MNIST(root_dir, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root_dir, train=False, download=True, transform=transform)

    colored_train_data = [
        get_single_color_mnist(Subset(mnist_train, r), RGB_tuples[i], i)
        for i, r in enumerate(train_ranges)
    ]
    colored_test_data = [
        get_single_color_mnist(Subset(mnist_test, r), RGB_tuples[i], i)
        for i, r in enumerate(test_ranges)
    ]

    train_images = torch.Tensor(np.concatenate([data[0] for data in colored_train_data]))
    train_labels = torch.tensor(np.concatenate([data[1] for data in colored_train_data]), dtype=torch.long)

    test_images = torch.Tensor(np.concatenate([data[0] for data in colored_test_data]))
    test_labels = torch.tensor(np.concatenate([data[1] for data in colored_test_data]), dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    return train_dataset, test_dataset


def get_single_rotation_mnist(dataset, rotation, domain):
    rotated_xs, ys, domains = [], [], []
    set_seed(0)
    for idx in range(len(dataset)):
        _x, _y = dataset[idx]
        rot_x = torchvision.transforms.functional.rotate(_x, rotation)
        rotated_x = np.array(rot_x).astype(np.float32)
        # Ensure the image is 4D (batch_size, channels, height, width)
        if len(rotated_x.shape) == 3:
            rotated_x = np.expand_dims(rotated_x, axis=0)
        rotated_xs.append(rotated_x)
        ys.append(_y)
        domains.append(domain)
    return np.concatenate(rotated_xs), np.array(ys)


def get_rotated_mnist(root_dir, transform, ratios=np.ones(NUM_COLORS)):
    set_seed(0)
    if len(ratios) == 1:
        train_ranges = [np.arange(TRAIN_N_SAMPLES)]
        test_ranges = [np.arange(TEST_N_SAMPLES)]
    else:
        ratios = np.array(ratios) / np.sum(ratios)
        cutoffs = np.concatenate([np.array([0]), np.cumsum(ratios)])
        cutoffs[-1] = 1.0
        cutoff_pairs = list(zip(cutoffs[:-1], cutoffs[1:]))
        train_ranges = [
            np.arange(int(TRAIN_N_SAMPLES * c_l), int(TRAIN_N_SAMPLES * c_r))
            for c_l, c_r in cutoff_pairs
        ]
        assert sum(len(r) for r in train_ranges) == TRAIN_N_SAMPLES
        test_ranges = [
            np.arange(int(TEST_N_SAMPLES * c_l), int(TEST_N_SAMPLES * c_r))
            for c_l, c_r in cutoff_pairs
        ]
        assert sum(len(r) for r in test_ranges) == TEST_N_SAMPLES

    mnist_train = torchvision.datasets.MNIST(root_dir, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root_dir, train=False, download=True, transform=transform)
    rotations = [np.random.randint(0, 360) for _ in range(len(train_ranges))]
    rotated_train_data = [
        get_single_rotation_mnist(Subset(mnist_train, r), rotations[i], i)
        for i, r in enumerate(train_ranges)
    ]
    rotated_test_data = [
        get_single_rotation_mnist(Subset(mnist_test, r), rotations[i], i)
        for i, r in enumerate(test_ranges)
    ]

    train_images = torch.Tensor(np.concatenate([data[0] for data in rotated_train_data]))
    train_labels = torch.tensor(np.concatenate([data[1] for data in rotated_train_data]), dtype=torch.long)

    test_images = torch.Tensor(np.concatenate([data[0] for data in rotated_test_data]))
    test_labels = torch.tensor(np.concatenate([data[1] for data in rotated_test_data]), dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    return train_dataset, test_dataset
