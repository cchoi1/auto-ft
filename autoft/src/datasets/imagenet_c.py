import os

import torch
from torch.utils.data import ConcatDataset

from .common import ImageFolderWithPaths
from .imagenet import ImageNet, CustomDataset

IMAGENET_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise",
                        "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise",
                        "snow", "zoom_blur"]

class ImageNetC(ImageNet):

    def __init__(
        self,
        preprocess,
        train,
        n_examples,
        severity=5,
        location = os.path.expanduser('~/data'),
        batch_size = 32,
        num_workers = 32,
        classnames = 'openai',
        custom = False,
    ):
        self.severity = severity
        super(ImageNetC, self).__init__(
            preprocess,
            train,
            n_examples,
            location,
            batch_size,
            num_workers,
            classnames,
            custom,
        )

    def populate_train(self):
        datasets = []
        for corruption in IMAGENET_CORRUPTIONS:
            traindir = os.path.join(self.location, 'ImageNet-C', corruption, str(self.severity))
            dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)
            if self.n_examples > -1:
                indices = list(range(self.n_examples))
                dataset = torch.utils.data.Subset(dataset, indices)
            datasets.append(dataset)

        self.dataset = ConcatDataset(datasets)

        if self.custom:
            custom_datasets = []
            for corruption in IMAGENET_CORRUPTIONS:
                traindir = os.path.join(self.location, 'ImageNet-C', corruption, str(self.severity))
                dataset = CustomDataset(root=traindir, transform=self.preprocess)
                if self.n_examples > -1:
                    indices = list(range(self.n_examples))
                    dataset = torch.utils.data.Subset(dataset, indices)
                custom_datasets.append(dataset)
            self.dataset = ConcatDataset(custom_datasets)

    def get_test_path(self):
        test_path = os.path.join(self.location, 'ImageNet-C', IMAGENET_CORRUPTIONS[0], str(self.severity))
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, 'ImageNet-C', IMAGENET_CORRUPTIONS[0], str(self.severity))
        return test_path

    def name(self):
        corruptions_str = "_".join(self.corruptions)
        return f'imagenet-c_{corruptions_str}_s{self.severity}'
