import os

import numpy as np
import torch
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class FMOW:
    test_subset = None

    def __init__(self,
                 preprocess,
                 train,
                 n_examples,
                 use_class_balanced=False,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=2,
                 subset='train',
                 classnames=None,
                 **kwargs):
        dataset = wilds.get_dataset(dataset='fmow', root_dir=location)
        if subset == 'train':
            self.dataset = dataset.get_subset('train', transform=preprocess)
            self.dataloader = get_train_loader("standard", self.dataset, num_workers=num_workers, batch_size=batch_size)
        elif "unlabeled" in subset:
            dataset = wilds.get_dataset(dataset='fmow', unlabeled=True, root_dir=location)
            self.dataset = dataset.get_subset('train_unlabeled', transform=preprocess)
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

        if subset in ['id_val', 'val']:
            self.class_to_indices = {}
            indices = self.dataset.indices
            for i in range(len(indices)):
                label = int(self.dataset[i][1])
                if label not in self.class_to_indices.keys():
                    self.class_to_indices[label] = []
                self.class_to_indices[label].append(i) # index relative to val set size for SubsetRandomSampler

        self.classnames = [
            "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
            "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership",
            "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution",
            "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
            "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
            "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility",
            "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
            "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track",
            "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall",
            "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank",
            "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal",
            "water_treatment_facility", "wind_farm", "zoo"
        ]

    def __len__(self):
        return len(self.dataset)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWTrain(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWUnlabeledTrain(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'unlabeled'
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWIDVal(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs["subset"] = "id_val"
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWOODVal(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs["subset"] = "val"
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWIDTest(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs["subset"] = "id_test"
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWOODTest(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs["subset"] = "test"
        super().__init__(*args, **kwargs)

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWKTrain(FMOW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler = self.get_train_sampler()

    def get_train_sampler(self):
        if hasattr(self.dataset, 'targets'):
            idxs = np.zeros(len(self.dataset.targets))
            target_array = np.array(self.dataset.targets)
            for c in range(len(self.classnames)):
                m = target_array == c
                n = len(idxs[m])
                arr = np.zeros(n)
                arr[:self.k()] = 1
                np.random.shuffle(arr)
                idxs[m] = arr

            idxs = idxs.astype('int')
            sampler = torch.utils.data.SubsetRandomSampler(np.where(idxs)[0])
            return sampler
        return None

    def k(self):
        return 1

ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"FMOW{k}Train"
    dyn_cls = type(cls_name, (FMOWKTrain, ), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls

class FMOWKIDVal(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs["subset"] = "id_val"
        super().__init__(*args, **kwargs)
        self.eval_sampler = self.get_eval_sampler()

    def get_eval_sampler(self):
        if hasattr(self.dataset, 'targets'):
            idxs = np.zeros(len(self.dataset.targets))
            target_array = np.array(self.dataset.targets)
            for c in range(len(self.classnames)):
                m = target_array == c
                n = len(idxs[m])
                arr = np.zeros(n)
                arr[:self.k()] = 1
                np.random.shuffle(arr)
                idxs[m] = arr

            idxs = idxs.astype('int')
            sampler = torch.utils.data.SubsetRandomSampler(np.where(idxs)[0])
            return sampler
        return None

    def k(self):
        return 1

for k in ks:
    cls_name = f"FMOWK{k}IDVal"
    dyn_cls = type(cls_name, (FMOWKIDVal, ), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls
