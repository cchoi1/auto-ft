import os

import numpy as np
import torch
import torchvision
from src.datasets.utils import SampledDataset
from src.datasets.utils import split_validation_set


CALTECH101_CLASSNAMES = [
    'off-center face', 'centered face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel',
    'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
    'side of a car', 'ceiling fan', 'cellphone', 'chair', 'chandelier', 'body of a cougar cat', 'face of a cougar cat',
    'crab', 'crayfish', 'crocodile', 'head of a  crocodile', 'cup', 'dalmatian', 'dollar bill', 'dolphin',
    'dragonfly', 'electric guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'head of a flamingo',
    'garfield', 'gerenuk', 'gramophone', 'grand piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis',
    'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin',
    'mayfly', 'menorah', 'metronome', 'minaret',     'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
    'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion',
    'sea horse', 'snoopy (cartoon beagle)',     'soccer ball', 'stapler', 'starfish', 'stegosaurus', 'stop sign',
    'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 'wheelchair', 'wild cat',
    'windsor chair', 'wrench', 'yin and yang symbol'
]

class Caltech101:
    def __init__(self,
                 preprocess,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=2,
                 subset='test',
                 **kwargs):

        self.n_examples = n_examples
        self.n_classes = 101

        if subset == 'train':
            self.data_location = os.path.join(location, 'caltech-101', 'train')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            print('len train dataset', len(self.dataset))
        elif 'val' in subset:
            self.data_location = os.path.join(location, 'caltech-101', 'val')
            save_path = os.path.join(self.data_location, 'val_split_indices.npy')
            dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            self.val_hopt_indices, self.val_early_stopping_indices = split_validation_set(dataset, save_path=save_path)
            if subset == 'val_hopt':
                self.dataset = torch.utils.data.Subset(dataset, self.val_hopt_indices)
                n_examples_per_class = self.n_examples // self.n_classes
                if self.n_examples > -1:
                    if use_class_balanced:
                        sampled_dataset = SampledDataset(self.dataset, "Caltech101ValHOpt", n_examples_per_class)
                        self.dataset = torch.utils.data.Subset(self.dataset, sampled_dataset.indices)
                    else:
                        indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                        self.dataset = torch.utils.data.Subset(self.dataset, indices)
            else:
                self.dataset = torch.utils.data.Subset(dataset, self.val_early_stopping_indices)
        elif subset == 'test':
            self.data_location = os.path.join(location, 'caltech-101', 'test')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        else:
            raise ValueError(f'Subset must be one of "train", "val", or "test".')

        print(f"Loading {subset} Data from ", self.data_location)

        self.classnames = CALTECH101_CLASSNAMES

    def __len__(self):
        return len(self.dataset)


class Caltech101Train(Caltech101):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)


class Caltech101ValHOpt(Caltech101):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_hopt'
        super().__init__(*args, **kwargs)


class Caltech101ValEarlyStopping(Caltech101):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_early_stopping'
        super().__init__(*args, **kwargs)


class Caltech101Test(Caltech101):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)
