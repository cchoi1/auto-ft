import os
import numpy as np
import torch
from torch.utils.data import Dataset

_TRAIN_IMAGES_FILENAME = 'train_images.npy'
_TRAIN_LABELS_FILENAME = 'train_labels.npy'
_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'
_CORRUPTIONS = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "impulse_noise", "motion_blur",
               "rotate", "scale", "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag"]

def check_exists(root, corruption):
    """Check if the dataset is present."""
    assert os.path.exists(os.path.join(root, corruption, _TRAIN_IMAGES_FILENAME)) and \
        os.path.exists(os.path.join(root, corruption, _TRAIN_LABELS_FILENAME)) and \
        os.path.exists(os.path.join(root, corruption, _TEST_IMAGES_FILENAME)) and \
        os.path.exists(os.path.join(root, corruption, _TEST_LABELS_FILENAME)), \
        f"Download the dataset first to {root}!"

class MNISTC(Dataset):
    """MNIST dataset with image-level corruptions."""
    def __init__(self, root, corruptions, train, transform=None):
        self.root = root
        self.corruptions = corruptions
        self.train = train
        self.transform = transform

        all_images, all_labels = [], []
        for corruption in self.corruptions:
            check_exists(root, corruption)
            images, labels = self._load_data(corruption)
            all_images.append(images)
            all_labels.append(labels)
        self.images = np.concatenate(all_images, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)

    def _load_data(self, corruption):
        if self.train:
            images_file = os.path.join(self.root, corruption, _TRAIN_IMAGES_FILENAME)
            labels_file = os.path.join(self.root, corruption, _TRAIN_LABELS_FILENAME)
        else:
            images_file = os.path.join(self.root, corruption, _TEST_IMAGES_FILENAME)
            labels_file = os.path.join(self.root, corruption, _TEST_LABELS_FILENAME)

        images = np.load(images_file)
        labels = np.load(labels_file)

        return images, labels
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label