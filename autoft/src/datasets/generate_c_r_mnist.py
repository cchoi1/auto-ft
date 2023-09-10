"""Code to create colored and rotated MNIST datasets."""

import colorsys
import os
import random

import torch
import numpy as np
import torchvision
from torch.utils.data import Subset

NUM_COLORS = 100
TRAIN_N_SAMPLES = 50000
TEST_N_SAMPLES = 10000

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def color_mnist_image(img, color, subsample=False):
    img = np.array(img).astype(np.uint8)
    if subsample:
        img = img[::2, ::2]
    np_digit_color = np.array(color).astype(np.uint8)
    np_digit = np.einsum("hw,c->hwc", img, np_digit_color)
    return np_digit

def get_single_color_mnist(dataset, color, domain):
    colored_xs, ys, domains = [], [], []
    for idx in range(len(dataset)):
        _x, _y = dataset[idx]
        _x = _x.squeeze(0)
        colored_x = color_mnist_image(_x, color=color)
        colored_xs.append(colored_x)
        ys.append(_y)
        domains.append(domain)

    return np.stack(colored_xs), np.array(ys)

def get_colored_mnist(root_dir, transform, ratios=np.ones(NUM_COLORS)):
    RGB_tuples = [(int(255 * r), int(255 * g), int(255 * b)) for r, g, b in [colorsys.hsv_to_rgb(i / NUM_COLORS, 1.0, 1.0) for i in range(NUM_COLORS)]]
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
    del mnist_train
    colored_test_data = [
        get_single_color_mnist(Subset(mnist_test, r), RGB_tuples[i], i)
        for i, r in enumerate(test_ranges)
    ]
    del mnist_test
    train_images = np.concatenate([data[0] for data in colored_train_data])
    train_labels = np.concatenate([data[1] for data in colored_train_data])
    del colored_train_data
    np.savez(os.path.join(root_dir, 'colored_mnist_train.npz'), images=train_images, labels=train_labels)
    del train_images, train_labels

    test_images = np.concatenate([data[0] for data in colored_test_data])
    test_labels = np.concatenate([data[1] for data in colored_test_data])
    del colored_test_data
    np.savez(os.path.join(root_dir, 'colored_mnist_test.npz'), images=test_images, labels=test_labels)
    del test_images, test_labels


def get_single_rotation_mnist(dataset, rotation, domain):
    rotated_xs, ys, domains = [], [], []
    for idx in range(len(dataset)):
        _x, _y = dataset[idx]
        pil_image = torchvision.transforms.ToPILImage()(_x)
        rot_x = torchvision.transforms.functional.rotate(pil_image, rotation)
        rotated_x = np.array(rot_x.convert('L')).astype(np.uint8)
        if len(rotated_x.shape) == 2:
            rotated_x = np.expand_dims(rotated_x, axis=0)

        rotated_xs.append(rotated_x)
        ys.append(_y)
        domains.append(domain)

    return np.stack(rotated_xs).transpose(0, 2, 3, 1), np.array(ys)


def get_rotated_mnist(root_dir, transform, ratios=np.ones(NUM_COLORS)):
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
    del mnist_train
    rotated_test_data = [
        get_single_rotation_mnist(Subset(mnist_test, r), rotations[i], i)
        for i, r in enumerate(test_ranges)
    ]
    del mnist_test

    train_images = np.concatenate([data[0] for data in rotated_train_data])
    train_labels = np.concatenate([data[1] for data in rotated_train_data])
    del rotated_train_data
    np.savez(os.path.join(root_dir, 'rotated_mnist_train.npz'), images=train_images, labels=train_labels)
    del train_images, train_labels

    test_images = np.concatenate([data[0] for data in rotated_test_data])
    test_labels = np.concatenate([data[1] for data in rotated_test_data])
    del rotated_test_data
    np.savez(os.path.join(root_dir, 'rotated_mnist_test.npz'), images=test_images, labels=test_labels)
    del test_images, test_labels


if __name__ == "__main__":
    set_seed(0)
    root_dir = "/iris/u/cchoi1/Data"
    colored_mnist_path = os.path.join(root_dir, "ColoredMNIST")
    os.makedirs(colored_mnist_path, exist_ok=True)
    get_colored_mnist(colored_mnist_path, transform=torchvision.transforms.ToTensor())
    rotated_mnist_path = os.path.join(root_dir, "RotatedMNIST")
    os.makedirs(rotated_mnist_path, exist_ok=True)
    get_rotated_mnist(rotated_mnist_path, transform=torchvision.transforms.ToTensor())