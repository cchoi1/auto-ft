import collections
import glob
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.datasets.laion import get_data
from src.models.utils import extract_from_data_parallel


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes-1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device, noscale):
    all_data = collections.defaultdict(list)
    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            image_encoder = image_encoder.to(inputs.device)
            features = image_encoder(inputs)
            if noscale:
                features = features / features.norm(dim=-1, keepdim=True)
            else:
                logit_scale = image_encoder.module.model.logit_scale
                features = logit_scale.exp() * features

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(args, is_train, image_encoder, dataset, device, cache_dir, noscale):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if cache_dir is not None:
        cache_dir = f'{cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = get_dataloader(dataset, is_train, args, image_encoder=None)
        data = get_features_helper(image_encoder, loader, device, noscale)
        if cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, args, is_train, image_encoder, dataset, device, cache_dir=None, noscale=True):
        self.data = get_features(args, is_train, image_encoder, dataset, device, cache_dir, noscale)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data

def collate_fn_for_cifar(batch):
    data, labels = zip(*batch)
    return torch.stack(data, 0), torch.tensor(labels).long()

def collate_fn_for_imagenet(batch):
    # Extract images, labels, features, and image_paths from the batch
    keys = batch[0].keys()
    batch_dict = {k : [] for k in keys}
    for k in keys:
        batch_dict[k] = [item[k] for item in batch]
    if "images" in keys:
        batch_dict["images"] = torch.stack(batch_dict["images"], 0)
    if "labels" in keys:
        batch_dict["labels"] = torch.tensor(batch_dict["labels"]).long()
    if "features" in keys:
        batch_dict["features"] = torch.stack(batch_dict["features"], 0)

    return batch_dict


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
            return data
        except StopIteration:
            # Reset the iterator when it reaches the end
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def __len__(self):
        return len(self.dataloader)


def create_dataloader(dataset, kwargs):
    """Helper function to create a DataLoader."""
    return DataLoader(dataset, **kwargs)


def get_dataloader(dataset, is_train, args, sampler=None, image_encoder=None):
    """
    Get a DataLoader for the given dataset.

    Args:
        dataset: Dataset object to be loaded.
        is_train: Boolean indicating if the dataset is for training.
        args: Arguments containing configurations.
        image_encoder: Optional image encoder for feature extraction.

    Returns:
        DataLoader for the given dataset.
    """
    kwargs = {"num_workers": args.workers, "pin_memory": True} if torch.cuda.is_available() else {}
    kwargs["batch_size"] = args.batch_size
    if args.distributed:
        kwargs["sampler"] = DistributedSampler(dataset)
    else:
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["shuffle"] = is_train
    if "ImageNet" in args.id:
        kwargs["collate_fn"] = collate_fn_for_imagenet
    elif "CIFAR" in args.id:
        kwargs["collate_fn"] = collate_fn_for_cifar

    if image_encoder is not None:
        kwargs["collate_fn"] = collate_fn_for_imagenet
        feature_dataset = FeatureDataset(args, is_train, image_encoder, dataset, args.device)
        return create_dataloader(feature_dataset, kwargs)

    # If the dataset is a wrapped dataset, retrieve the underlying dataset
    if hasattr(dataset, 'dataset'):
        inner_dataset = dataset.dataset
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        inner_dataset = dataset
    else:
        inner_dataset = dataset

    return create_dataloader(inner_dataset, kwargs)

def get_autoft_dataloaders(args, all_datasets):
    if args.ft_data is not None and args.k is None:
        id_dataloader = all_datasets["id"]["train_ft"].dataloader
    elif args.ft_data is not None and args.k is not None:
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)

    if args.k is not None:
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=None)
    else:
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=None)
    ood_hp_dataloader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=True, args=args, image_encoder=None)
    if args.unlabeled_id is not None:
        unlabeled_dataloader = all_datasets["id_unlabeled"].dataloader
    else:
        unlabeled_dataloader = None

    dataloaders = {"id": id_dataloader, "id_val": id_val_dataloader, "ood_hp": ood_hp_dataloader, "unlabeled": unlabeled_dataloader}
    return dataloaders