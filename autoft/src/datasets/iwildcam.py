import os
import pandas as pd
import json
import numpy as np
import pathlib

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset


def get_mask_non_empty(dataset):
    metadf = pd.read_csv(dataset._data_dir / 'metadata.csv')
    filename = os.path.expanduser(dataset._data_dir / 'iwildcam2020_megadetector_results.json')
    with open(filename, 'r') as f:
        md_data = json.load(f)
    id_to_maxdet = {x['id']: x['max_detection_conf'] for x in md_data['images']}
    threshold = 0.95
    mask_non_empty = [id_to_maxdet[x] >= threshold for x in metadf['image_id']]
    return mask_non_empty


def get_nonempty_subset(dataset, split, frac=1.0, transform=None):
    if split not in dataset.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")
    split_mask = dataset.split_array == dataset.split_dict[split]

    # intersect split mask with non_empty. here is the only place this fn differs
    # from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py#L56
    mask_non_empty = get_mask_non_empty(dataset)
    split_mask = split_mask & mask_non_empty

    split_idx = np.where(split_mask)[0]
    if frac < 1.0:
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
    subset = WILDSSubset(dataset, split_idx, transform)
    return subset


class IWildCam:
    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 remove_non_empty=False,
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 subset='train'):
        dataset = wilds.get_dataset(dataset='iwildcam', root_dir=location)
        if subset == 'train':
            self.dataset = dataset.get_subset('train', transform=preprocess)
            self.dataloader = get_train_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif "unlabeled" in subset:
            dataset = wilds.get_dataset(dataset='iwildcam', unlabeled=True, root_dir=location)
            self.dataset = dataset.get_subset('extra_unlabeled', transform=preprocess)
            self.dataloader = get_train_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif subset == 'val':
            self.dataset = dataset.get_subset('val', transform=preprocess)
            self.dataloader = get_eval_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif subset == 'id_val':
            self.dataset = dataset.get_subset('id_val', transform=preprocess)
            self.dataloader = get_eval_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif subset == 'id_test':
            self.dataset = dataset.get_subset('id_test', transform=preprocess)
            self.dataloader = get_eval_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif subset == 'test':
            self.dataset = dataset.get_subset('test', transform=preprocess)
            self.dataloader = get_eval_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)

        labels_csv = pathlib.Path(__file__).parent / 'iwildcam_metadata' / 'labels.csv'
        df = pd.read_csv(labels_csv)
        df = df[df['y'] < 99999]
        
        self.classnames = [s.lower() for s in list(df['english'])]

    def __len__(self):
        return len(self.dataset)

    def post_loop_metrics(self, labels, preds, metadata, args):
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class IWildCamTrain(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)

class IWildCamUnlabeledTrain(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'unlabeled'
        super().__init__(*args, **kwargs)

class IWildCamIDVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_val'
        super().__init__(*args, **kwargs)

class IWildCamIDTest(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)

class IWildCamOODVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)

class IWildCamOODTest(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)


# class IWildCamNonEmpty(IWildCam):
#     def __init__(self, *args, **kwargs):
#         kwargs['subset'] = 'train'
#         super().__init__(*args, **kwargs)
#
#
# class IWildCamIDNonEmpty(IWildCam):
#     def __init__(self, *args, **kwargs):
#         kwargs['subset'] = 'id_test'
#         super().__init__(*args, **kwargs)
#
#
# class IWildCamOODNonEmpty(IWildCam):
#     def __init__(self, *args, **kwargs):
#         kwargs['subset'] = 'test'
#         super().__init__(*args, **kwargs)
