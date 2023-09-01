import numpy as np
from torch.utils.data import Dataset

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
        unique_classes = np.unique(targets)

        sampled_indices = []

        for cls in unique_classes:
            class_indices = np.where(targets == cls)[0]

            # Check if the number of samples per class is not greater than the available samples in the dataset
            if num_samples_per_class > len(class_indices):
                raise ValueError(
                    f"Number of samples per class for class {cls} is greater than the available samples in the dataset.")

            # Subsample the appropriate number of instances from the class
            sampled_indices_class = np.random.choice(class_indices, num_samples_per_class, replace=False)
            sampled_indices.append(sampled_indices_class)

        # Combine the indices from each class and shuffle them
        indices = np.concatenate(sampled_indices)
        np.random.shuffle(indices)

        return indices.tolist()
