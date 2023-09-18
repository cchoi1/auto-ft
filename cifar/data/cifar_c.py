from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
import torch
from typing import List


_CORRUPTIONS = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "impulse_noise", "motion_blur",
                "rotate", "scale", "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag"]


class CIFAR10C(Dataset):
    def __init__(self,
                 root_dir: str,
                 corruptions: List[str],
                 transform = None,
                 n_examples: int = 10000,
                 severity: int = 5,
                 shuffle: bool = False):

        assert 1 <= severity <= 5, "Severity level must be between 1 and 5."
        assert 0 <= n_examples <= 10000, "Number of examples must be between 0 and 10000."
        self.n_total_cifar = 10000
        self.root_dir = Path(root_dir)
        self.n_examples = n_examples
        self.severity = severity
        self.corruptions = corruptions
        self.shuffle = shuffle
        self.transform = transform

        if not self.root_dir.exists():
            os.makedirs(self.root_dir)

        data_root_dir = self.root_dir / 'CIFAR-10-C'
        # Load labels
        labels_path = data_root_dir / 'labels.npy'
        if not os.path.isfile(labels_path):
            raise ValueError("Labels are missing, try to re-download them.")
        self.labels = np.load(labels_path)

        self.data, self.targets = self.load_data()

    def load_data(self):
        x_test_list, y_test_list = [], []
        n_pert = len(self.corruptions)
        for corruption in self.corruptions:
            corruption_file_path = self.root_dir / (corruption + '.npy')
            if not corruption_file_path.is_file():
                raise ValueError(f"{corruption} file is missing, try to re-download it.")

            images_all = np.load(corruption_file_path)
            images = images_all[(self.severity - 1) * self.n_total_cifar:self.severity * self.n_total_cifar]
            n_img = int(np.ceil(self.n_examples / n_pert))
            x_test_list.append(images[:n_img])
            y_test_list.append(self.labels[:n_img])

        x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
        if self.shuffle:
            rand_idx = np.random.permutation(np.arange(len(x_test)))
            x_test, y_test = x_test[rand_idx], y_test[rand_idx]

        x_test = np.transpose(x_test, (0, 3, 1, 2))
        x_test = x_test.astype(np.float32) / 255
        x_test, y_test = torch.tensor(x_test)[:self.n_examples], torch.tensor(y_test)[:self.n_examples]

        return x_test, y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
