import json
import os

import numpy as np
from torch.utils.data import Dataset

def get_ood_datasets(dataset, num_labeled_examples, num_unlabeled_examples):
    targets = np.array([target for _, target in dataset])
    unique_classes = np.unique(targets)

    num_labeled_per_class = num_labeled_examples // len(unique_classes)
    if num_unlabeled_examples is not None:
        num_unlabeled_per_class = num_unlabeled_examples // len(unique_classes)
    else:
        num_unlabeled_per_class = 0

    sampled_indices_labeled = []
    sampled_indices_unlabeled = []
    for cls in unique_classes:
        class_indices = np.where(targets == cls)[0]
        if num_labeled_per_class + num_unlabeled_per_class > len(class_indices):
            raise ValueError(f"Total number of required samples for class {cls} exceeds available samples.")
        sampled_class_indices = np.random.choice(class_indices, num_labeled_per_class + num_unlabeled_per_class, replace=False)

        sampled_class_indices_labeled = sampled_class_indices[:num_labeled_per_class]
        sampled_class_indices_unlabeled = sampled_class_indices[num_labeled_per_class:]
        sampled_indices_labeled.append(sampled_class_indices_labeled)
        sampled_indices_unlabeled.append(sampled_class_indices_unlabeled)

    indices_labeled = np.concatenate(sampled_indices_labeled)
    np.random.shuffle(indices_labeled)
    indices_unlabeled = np.concatenate(sampled_indices_unlabeled)
    np.random.shuffle(indices_unlabeled)

    subset_labeled = SampledDataset(dataset, num_samples_per_class=num_labeled_per_class)
    subset_unlabeled = UnlabeledDatasetWrapper(SampledDataset(dataset, num_samples_per_class=num_unlabeled_per_class))
    subset_labeled.indices = indices_labeled.tolist()
    subset_unlabeled.indices = indices_unlabeled.tolist()

    return subset_labeled, subset_unlabeled

class SampledDataset(Dataset):
    """
    Dataset class that samples a fixed number of instances from each class in the original dataset.
    """

    def __init__(self, dataset, dataset_name, num_samples_per_class, save_dir="/home/carolinechoi/robust-ft/data"):
        self.dataset = dataset
        os.makedirs(save_dir, exist_ok=True)
        indices_file = f"{dataset_name}_sample_indices_{num_samples_per_class}.json"
        self.save_path = os.path.join(save_dir, indices_file)
        print('num_samples_per_class', num_samples_per_class)

        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.indices = json.load(f)
            print("Loading class-balanced indices from file.")
            self.indices = [int(index) for index in self.indices]
        else:
            print("Generating class-balanced indices.")
            self.indices = self.get_indices(num_samples_per_class)
            with open(self.save_path, 'w') as f:
                json.dump(self.indices, f)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def get_indices(self, num_samples_per_class):
        # Check the format of the first item to determine how to extract labels
        first_item = self.dataset[0]
        if isinstance(first_item, tuple):
            # Old format: Tuple format (image, label)
            targets = np.array([label for _, label in self.dataset])
        elif isinstance(first_item, dict) and 'labels' in first_item:
            # New format: Dictionary with keys 'images' and 'labels'
            targets = np.array([item['labels'] for item in self.dataset])
        else:
            raise ValueError("Unsupported dataset format.")

        unique_classes = np.unique(targets)

        sampled_indices = []

        for cls in unique_classes:
            class_indices = np.where(targets == cls)[0]

            if num_samples_per_class > len(class_indices):
                raise ValueError(
                    f"Number of samples per class for class {cls} is greater than the available samples in the dataset.")

            sampled_indices_class = np.random.choice(class_indices, num_samples_per_class, replace=False)
            sampled_indices.append(sampled_indices_class)

        indices = np.concatenate(sampled_indices)
        np.random.shuffle(indices)

        return indices.tolist()


class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, labeled_dataset):
        self.labeled_dataset = labeled_dataset

    def __getitem__(self, index):
        image, _ = self.labeled_dataset[index]
        return image, -1

    def __len__(self):
        return len(self.labeled_dataset)


def split_validation_set(dataset, split_ratio=0.5, save_path=None):
    """
    Split dataset in a class-balanced manner. Loads from saved split indices if available
    :param dataset: PyTorch dataset to split.
    :param split_ratio: Fraction of data for the first split (e.g., 0.5 means a 50-50 split).
    :param save_path: Path to save/load indices for reproducibility.
    :return: Two lists of indices representing the two splits.
    """

    # Check if indices have been saved earlier
    if save_path and os.path.exists(save_path):
        all_indices = np.load(save_path, allow_pickle=True)
        return all_indices.item().get("val_hopt"), all_indices.item().get("val_early_stopping")

    # Split each class in the dataset
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    split_1_indices = []
    split_2_indices = []
    for c in classes:
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)

        split_point = int(len(class_indices) * split_ratio)
        split_1_indices.extend(class_indices[:split_point])
        split_2_indices.extend(class_indices[split_point:])

    # Save the indices for reproducibility if a save path is provided
    if save_path:
        np.save(save_path, {"val_hopt": split_1_indices, "val_early_stopping": split_2_indices})

    return split_1_indices, split_2_indices

