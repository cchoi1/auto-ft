from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


class SampledDataset(Dataset):
    """
    Dataset class that samples a fixed number of instances from each class in the original dataset.
    """
    def __init__(self, original_dataset, num_samples_per_class):
        self.original_dataset = original_dataset
        self.indices = self.get_indices(num_samples_per_class)

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def get_indices(self, num_samples_per_class):
        targets = np.array([target for _, target in self.original_dataset])
        class0_indices = np.where(targets == 0)[0]
        class1_indices = np.where(targets == 1)[0]

        # Check if the number of samples per class is not greater than the available samples in the dataset
        if num_samples_per_class > len(class0_indices) or num_samples_per_class > len(class1_indices):
            raise ValueError("Number of samples per class is greater than the available samples in the dataset.")

        # Subsample the appropriate number of instances from each class
        sampled_indices_class0 = np.random.choice(class0_indices, num_samples_per_class, replace=False)
        sampled_indices_class1 = np.random.choice(class1_indices, num_samples_per_class, replace=False)

        # Combine the indices from each class and shuffle them
        indices = np.concatenate([sampled_indices_class0, sampled_indices_class1])
        np.random.shuffle(indices)

        return indices.tolist()