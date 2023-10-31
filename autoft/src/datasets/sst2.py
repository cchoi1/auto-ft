import os

import torch
import torchvision
from src.datasets.utils import split_validation_set


class sst2:
    def __init__(self,
                 preprocess,
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

        # Load data based on the subset argument
        if subset == 'train':
            if self.k is not None:
                self.data_location = os.path.join(location, 'sst2', f'train_shot_{self.k}')
            else:
                self.data_location = os.path.join(location, 'sst2', 'train')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        elif 'val' in subset:
            self.data_location = os.path.join(location, 'sst2', 'val')
            save_path = os.path.join(self.data_location, 'val_split_indices.npy')
            dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            self.val_hopt_indices, self.val_early_stopping_indices = split_validation_set(dataset, save_path=save_path)
            if subset == 'val_hopt':
                self.dataset = torch.utils.data.Subset(dataset, self.val_hopt_indices)
            else:
                self.dataset = torch.utils.data.Subset(dataset, self.val_early_stopping_indices)
        elif subset == 'test':
            self.data_location = os.path.join(location, 'sst2', 'test')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        else:
            raise ValueError(f'Subset must be one of "train", "val_hopt", "val_early_stopping", or "test".')

        print(f"Loading {subset} Data from ", self.data_location)

        self.classnames = ['negative', 'positive']

    def __len__(self):
        return len(self.dataset)


class sst2Train(sst2):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)

class sst2ValHOpt(sst2):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_hopt'
        super().__init__(*args, **kwargs)


class sst2ValEarlyStopping(sst2):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_early_stopping'
        super().__init__(*args, **kwargs)


class sst2Test(sst2):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)
