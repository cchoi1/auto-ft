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
        use_class_balanced,
        severity=5,
        location = os.path.expanduser('~/data'),
        batch_size = 32,
        num_workers = 32,
        classnames = 'openai',
        custom = False,
    ):
        self.use_class_balanced = use_class_balanced
        self.severity = severity
        super(ImageNetC, self).__init__(
            preprocess,
            train,
            n_examples,
            use_class_balanced,
            location,
            batch_size,
            num_workers,
            classnames,
            custom,
        )

    def populate_train(self):
        if self.use_class_balanced:
            datasets = []
            for corruption in IMAGENET_CORRUPTIONS:
                traindir = os.path.join(self.location, 'ImageNet-C', corruption, str(self.severity))
                dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)

                if self.n_examples > -1:
                    num_samples_per_class = self.n_examples // (len(IMAGENET_CORRUPTIONS) * self.num_classes)
                    dataset = SampledDataset(dataset, self.__str__(), num_samples_per_class=num_samples_per_class)
            datasets.append(dataset)
            self.dataset = ConcatDataset(datasets)
        else:
            traindir = os.path.join(self.location, 'ImageNet-C', IMAGENET_CORRUPTIONS[0], str(self.severity))
            self.dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)
            if self.n_examples > -1:
                rand_idxs = torch.randperm(len(self.dataset))[:self.n_examples]
                self.dataset = torch.utils.data.Subset(self.dataset, rand_idxs)

        if self.custom:
            if self.use_class_balanced:
                custom_datasets = []
                for corruption in IMAGENET_CORRUPTIONS:
                    traindir = os.path.join(self.location, 'ImageNet-C', corruption, str(self.severity))
                    dataset = CustomDataset(root=traindir, transform=self.preprocess)
                    if self.n_examples > -1:
                        num_samples_per_class = self.n_examples // (len(IMAGENET_CORRUPTIONS) * self.num_classes)
                        dataset = SampledDataset(dataset, self.__str__(), num_samples_per_class=num_samples_per_class)
                    custom_datasets.append(dataset)
                self.dataset = ConcatDataset(custom_datasets)
            else:
                traindir = os.path.join(self.location, 'ImageNet-C', IMAGENET_CORRUPTIONS[0], str(self.severity))
                self.dataset = CustomDataset(root=traindir, transform=self.preprocess)
                if self.n_examples > -1:
                    rand_idxs = torch.randperm(len(self.dataset))[:self.n_examples]
                    self.dataset = torch.utils.data.Subset(self.dataset, rand_idxs)

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
