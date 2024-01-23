import glob
import os

import numpy as np
import torch
import torchvision
from PIL import Image

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames


class CustomDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transforms = transform
        self.class_list = sorted(os.listdir(root))
        self.img_list = []
        self.class_len_list = []
        for i, c in enumerate(self.class_list):
            root_child = os.path.join(root, c)
            self.img_list.append(sorted(glob.glob(root_child + "/*")))
            self.class_len_list.append(len(self.img_list[-1]))

    def __len__(self):
        total_len = 0
        for i, c in enumerate(self.class_list):
            total_len += len(self.img_list[i])
        return total_len

    def __getitem__(self, idx):
        batch_img = []
        # batch_y = []
        for i, c in enumerate(self.class_list):
            rand_idx = np.random.randint(0, self.class_len_list[i])
            img_name = self.img_list[i][rand_idx]
            image = self.transforms(Image.open(img_name))
            batch_img.append(image)
            # batch_y.append(i)

        batch_img = torch.stack(batch_img, dim=0)

        return batch_img


class ImageNet:
    def __init__(
            self,
            preprocess,
            train,
            n_examples,
            use_class_balanced=False,
            location=os.path.expanduser('~/data'),
            batch_size=32,
            num_workers=32,
            classnames='openai',
            custom=False,
    ):
        self.preprocess = preprocess
        self.train = train
        self.num_classes = 1000
        self.n_examples = n_examples
        self.use_class_balanced = use_class_balanced
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.custom = custom
        if self.train:
            self.populate_train()
        else:
            self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, 'ImageNet/ILSVRC/Data/CLS-LOC', 'train')
        self.dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)

    def populate_test(self):
        self.dataset = self.get_test_dataset()

    def get_test_path(self):
        test_path = os.path.join(self.location, 'ImageNet/ILSVRC/Data/CLS-LOC', 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, 'ImageNet/ILSVRC/Data/CLS-LOC', 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(),
                                    transform=self.preprocess)

    def name(self):
        return 'imagenet'

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "ImageNet"


class ImageNetTrain(ImageNet):
    def get_test_dataset(self):
        pass


class ImageNetK(ImageNet):
    def get_train_sampler(self):
        idxs = np.zeros(len(self.dataset.targets))
        target_array = np.array(self.dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler

    def __str__(self):
        return f"ImageNet{self.k()}"


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)


class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask(
        )
        self.classnames = [self.classnames[i] for i in class_sublist]
        self.num_classes = len(self.classnames)

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask(
        )
        idx_subsample_list = [
            range(x * 50, (x + 1) * 50) for x in self.class_sublist
        ]
        idx_subsample_list = sorted(
            [item for sublist in idx_subsample_list for item in sublist])

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [
            self.class_sublist.index(int(label)) for label in labels
        ]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"ImageNet{k}"
    dyn_cls = type(cls_name, (ImageNetK,), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls
