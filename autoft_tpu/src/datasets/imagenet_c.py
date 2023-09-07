import os

import torch
from torch.utils.data import ConcatDataset

from .common import ImageFolderWithPaths
from .imagenet import ImageNet, CustomDataset
from .utils import SampledDataset

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

            # If n_examples is specified, then balance the dataset per corruption
            if self.n_examples > -1:
                # Use the total number of examples per corruption and then further divide by number of classes for class-balancing
                num_samples_per_class = self.n_examples // (len(IMAGENET_CORRUPTIONS) * self.num_classes)
                dataset = SampledDataset(dataset, num_samples_per_class=num_samples_per_class)
            datasets.append(dataset)
        self.dataset = ConcatDataset(datasets)

        if self.custom:
            custom_datasets = []
            for corruption in IMAGENET_CORRUPTIONS:
                traindir = os.path.join(self.location, 'ImageNet-C', corruption, str(self.severity))
                dataset = CustomDataset(root=traindir, transform=self.preprocess)
                if self.n_examples > -1:
                    num_samples_per_class = self.n_examples // (len(IMAGENET_CORRUPTIONS) * self.num_classes)
                    dataset = SampledDataset(dataset, num_samples_per_class=num_samples_per_class)
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

    def __str__(self):
        return "ImageNetC"
