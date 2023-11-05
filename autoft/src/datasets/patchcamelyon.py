import os

import numpy as np
import torch
import torchvision
from src.datasets.utils import SampledDataset
from src.datasets.utils import split_validation_set


class PatchCamelyon:
    def __init__(self,
                 preprocess,
                 train=None,
                 n_examples=-1,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=2,
                 subset='test',
                 classnames=None,
                 custom=False,
                 k=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        self.n_examples = n_examples
        self.n_classes = 2

        # Load data based on the subset argument
        if subset == 'train':
            if self.k is not None:
                self.data_location = os.path.join(location, 'patchcamelyon', f'train_shot_{self.k}')
            else:
                self.data_location = os.path.join(location, 'patchcamelyon', 'train')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        elif 'val' in subset:
            self.data_location = os.path.join(location, 'patchcamelyon', 'val')
            save_path = os.path.join(self.data_location, 'val_split_indices.npy')
            dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            self.val_hopt_indices, self.val_early_stopping_indices = split_validation_set(dataset, save_path=save_path)
            if subset == 'val_hopt':
                self.dataset = torch.utils.data.Subset(dataset, self.val_hopt_indices)
                if self.n_examples > -1:
                    if use_class_balanced:
                        n_examples_per_class = self.n_examples // self.n_classes
                        sampled_dataset = SampledDataset(self.dataset, "PatchCamelyonValHopt", n_examples_per_class)
                        self.dataset = torch.utils.data.Subset(self.dataset, sampled_dataset.indices)
                    else:
                        indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                        self.dataset = torch.utils.data.Subset(self.dataset, indices)
            else:
                self.dataset = torch.utils.data.Subset(dataset, self.val_early_stopping_indices)
        elif subset == 'test':
            self.data_location = os.path.join(location, 'patchcamelyon', 'test')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        else:
            raise ValueError(f'Subset must be one of "train", "val_hopt", "val_early_stopping", or "test".')

        print(f"Loading {subset} Data from ", self.data_location)

        # Sample classnames for the PatchCamelyon dataset
        self.classnames = [
            'lymph node',
            'lymph node containing metastatic tumor tissue'
        ]

    def __len__(self):
        return len(self.dataset)

class PatchCamelyonTrain(PatchCamelyon):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)

class PatchCamelyonValHOpt(PatchCamelyon):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_hopt'
        super().__init__(*args, **kwargs)

class PatchCamelyonValEarlyStopping(PatchCamelyon):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_early_stopping'
        super().__init__(*args, **kwargs)

class PatchCamelyonTest(PatchCamelyon):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)
