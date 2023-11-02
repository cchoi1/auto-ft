import os

import numpy as np
import torch
import torchvision
from src.datasets.utils import SampledDataset
from src.datasets.utils import split_validation_set


class Flowers102:
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
        if subset == 'train':
            self.data_location = os.path.join(location, 'flowers102', 'train')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            print('len train dataset', len(self.dataset))
        elif 'val' in subset:
            self.data_location = os.path.join(location, 'flowers102', 'val')
            save_path = os.path.join(self.data_location, 'val_split_indices.npy')
            dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
            self.val_hopt_indices, self.val_early_stopping_indices = split_validation_set(dataset, save_path=save_path)
            if subset == 'val_hopt':
                self.dataset = torch.utils.data.Subset(dataset, self.val_hopt_indices)
                if self.n_examples > -1:
                    if use_class_balanced:
                        sampled_dataset = SampledDataset(self.dataset, "Flowers102ValHOpt", n_examples)
                        self.dataset = torch.utils.data.Subset(self.dataset, sampled_dataset.indices)
                    else:
                        indices = np.random.choice(len(self.dataset), n_examples, replace=False)
                        self.dataset = torch.utils.data.Subset(self.dataset, indices)
            else:
                self.dataset = torch.utils.data.Subset(dataset, self.val_early_stopping_indices)
        elif subset == 'test':
            self.data_location = os.path.join(location, 'flowers102', 'test')
            self.dataset = torchvision.datasets.ImageFolder(root=self.data_location, transform=preprocess)
        else:
            raise ValueError(f'Subset must be one of "train", "val", or "test".')

        print(f"Loading {subset} Data from ", self.data_location)


        self.classnames = [
            'air plant', 'alpine sea holly', 'anthurium', 'artichoke',
            'azalea', 'balloon flower', 'barbeton daisy', 'bearded iris',
            'bee balm', 'bird of paradise', 'bishop of llandaff',
            'black-eyed susan', 'blackberry lily', 'blanket flower',
            'bolero deep blue', 'bougainvillea', 'bromelia', 'buttercup',
            'californian poppy', 'camellia', 'canna lily', 'canterbury bells',
            'cape flower', 'carnation', 'cautleya spicata', 'clematis',
            "colt's foot", 'columbine', 'common dandelion', 'corn poppy',
            'cyclamen', 'daffodil', 'desert-rose', 'english marigold',
            'fire lily', 'foxglove', 'frangipani', 'fritillary',
            'garden phlox', 'gaura', 'gazania', 'geranium',
            'giant white arum lily', 'globe flower', 'globe thistle',
            'grape hyacinth', 'great masterwort', 'hard-leaved pocket orchid',
            'hibiscus', 'hippeastrum', 'japanese anemone', 'king protea',
            'lenten rose', 'lotus', 'love in the mist', 'magnolia', 'mallow',
            'marigold', 'mexican aster', 'mexican petunia', 'monkshood',
            'moon orchid', 'morning glory', 'orange dahlia', 'osteospermum',
            'oxeye daisy', 'passion flower', 'pelargonium', 'peruvian lily',
            'petunia', 'pincushion flower', 'pink and yellow dahlia',
            'pink primrose', 'poinsettia', 'primula',
            'prince of wales feathers', 'purple coneflower', 'red ginger',
            'rose', 'ruby-lipped cattleya', 'siam tulip', 'silverbush',
            'snapdragon', 'spear thistle', 'spring crocus', 'stemless gentian',
            'sunflower', 'sweet pea', 'sweet william', 'sword lily',
            'thorn apple', 'tiger lily', 'toad lily', 'tree mallow',
            'tree poppy', 'trumpet creeper', 'wallflower', 'water lily',
            'watercress', 'wild pansy', 'windflower', 'yellow iris'
        ]

    def __len__(self):
        return len(self.dataset)


class Flowers102Train(Flowers102):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)


class Flowers102ValHOpt(Flowers102):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_hopt'
        super().__init__(*args, **kwargs)


class Flowers102ValEarlyStopping(Flowers102):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val_early_stopping'
        super().__init__(*args, **kwargs)


class Flowers102Test(Flowers102):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)