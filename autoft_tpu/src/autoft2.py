import gc
import os
import time

import numpy as np
import optuna
import src.datasets as datasets
import src.losses as losses
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from src.args import parse_arguments
from src.datasets.common import collate_fn_for_imagenet, collate_fn_for_cifar, FeatureDataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.utils import UnlabeledDatasetWrapper
from src.logger import setup_logging
from src.models.utils import cosine_lr
from src.models.utils import initialize_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_datasets(args, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    id_dataset = id_dataset_class(preprocess_fn, train=True, n_examples=args.num_id_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    ood_dataset_class = getattr(datasets, args.ood)
    n_examples = -1 if args.num_ood_unlabeled_examples is not None else args.num_ood_hp_examples
    ood_dataset_kwargs = {"preprocess": preprocess_fn, "train": True, "n_examples": n_examples,
                          "location": args.data_location, "batch_size": args.batch_size, "num_workers": args.workers}
    if args.ood == args.id: # Use the test split of the ID distribution as OOD
        ood_dataset_kwargs["train"] = False
    else:
        if args.ood == "CIFAR10C":
            ood_dataset_kwargs["severity"] = args.severity
    ood_dataset = ood_dataset_class(**ood_dataset_kwargs)
    if args.num_ood_unlabeled_examples is not None:
        # ood_labeled_dataset, ood_unlabeled_dataset = get_ood_datasets(
        #     ood_dataset.dataset,
        #     args.num_ood_hp_examples,
        #     args.num_ood_unlabeled_examples
        # )
        # ood_subset_for_hp = torch.utils.data.ConcatDataset([ood_labeled_dataset, ood_unlabeled_dataset])
        ood_subset_for_hp = UnlabeledDatasetWrapper(ood_dataset)
        xm.master_print("Using unlabeled OOD data...")
    else:
        ood_subset_for_hp = ood_dataset

    all_datasets = {"id": id_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    return all_datasets

def get_sampler(dataset, train):
    """Helper function to create a sampler."""
    if xm.xrt_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=train)
    else:
        sampler = None
    return sampler


def get_loss_weights(hyperparams, loss_type, num_losses=None):
    if loss_type == "LayerwiseLoss":
        loss_weight_keys = [k for k in hyperparams.keys() if "lossw" in k]
        layer_idx = 0
        layer_loss_weights = []
        loss_weights = []
        for k in loss_weight_keys:
            if int(k.split("_")[1]) == layer_idx:
                layer_loss_weights.append(hyperparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hyperparams[k]]
                layer_idx += 1
    else:
        assert num_losses is not None
        loss_weights = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(num_losses)])
    return loss_weights

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
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "persistent_workers": args.persistent_workers,
              "prefetch_factor": args.prefetch_factor, "pin_memory": True}
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["sampler"] = get_sampler(dataset, is_train)
    if is_train and kwargs["sampler"] is None:
        kwargs["shuffle"] = True
    else:
        kwargs["shuffle"] = False
    if "ImageNet" in args.id:
        kwargs["collate_fn"] = collate_fn_for_imagenet
    elif "CIFAR" in args.id:
        kwargs["collate_fn"] = collate_fn_for_cifar
    if image_encoder is not None:
        kwargs["collate_fn"] = collate_fn_for_imagenet
        dataset = FeatureDataset(args, is_train, image_encoder, dataset, args.device)
    elif hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    kwargs["dataset"] = dataset
    dataloader = DataLoader(**kwargs)
    device = xm.xla_device()
    dataloader = pl.MpDeviceLoader(dataloader, device, loader_prefetch_size=args.loader_prefetch_size, device_prefetch_size=args.device_prefetch_size)
    return dataloader

class HyperparameterSpace:
    def __init__(self, model, loss_type, num_losses=None):
        self.model = model
        self.loss_type = loss_type
        self.num_losses = num_losses
        if loss_type == "LayerwiseLoss":
            assert num_losses is not None

    def _base_space(self, trial, prefix):
        return {
            f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-5, 1e-2, log=True),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0),
            **{f"{prefix}lossw_{i}": trial.suggest_float(f"{prefix}lossw_{i}", 1e-5, 1e-2, log=True) for i in range(self.num_losses)}
        }

    def build_learned_loss_space(self, trial):
        hparams = self._base_space(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        return hparams

    def build_layerwise_loss_space(self, trial):
        hparams = {}
        layer_idx = 0
        for name, module in self.model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    hparams.update(self._base_space(trial, f"{layer_idx}_"))
                    layer_idx += 1
                # Classification head of the model
                hparams.update(self._base_space(trial, f"{layer_idx}_"))
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        return hparams

    def build_space(self, trial):
        if self.loss_type == "LayerwiseLoss":
            return self.build_layerwise_loss_space(trial)
        else:
            assert self.num_losses is not None
            return self.build_learned_loss_space(trial)
        del self.model
        gc.collect()

def create_optimizer(model, hyperparams, loss_type):
    if loss_type == "LayerwiseLoss":
        layerwise_params = []
        layer_idx = 0
        # Extract layers from the image_encoder (CLIPEncoder) of the model
        for name, module in model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    params_for_layer = {
                        'params': sub_module.parameters(),
                        'lr': hyperparams[f"lr_{layer_idx}"],
                        'weight_decay': hyperparams[f"wd_{layer_idx}"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

        # Classification head of the model
        params_for_layer = {
            'params': model.classification_head.parameters(),
            'lr': hyperparams[f"lr_{layer_idx}"],
            'weight_decay': hyperparams[f"wd_{layer_idx}"]
        }
        layerwise_params.append(params_for_layer)
        optimizer = torch.optim.AdamW(layerwise_params)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
    return optimizer

def objective(trial, device, args):
    model, preprocess_fn = initialize_model(args)
    model = model.to(device)
    hspace = HyperparameterSpace(model, args.loss_type, args.num_losses)
    hparams = hspace.build_space(trial)
    loss_weights = get_loss_weights(hparams, args.loss_type, args.num_losses)
    model_params = [p for p in model.parameters()]
    # loss_fn = getattr(losses, args.loss_type)(loss_weights, model_params)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, hparams, args.loss_type)
    if args.freeze_encoder:
        input_key = 'features'
    else:
        input_key = 'images'
    all_datasets = get_datasets(args, preprocess_fn)
    train_loader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
    test_loader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=False, args=args, image_encoder=None)
    num_batches = len(train_loader)
    total_steps = int(num_batches / args.accumulation_steps) * args.ft_epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)

    for step, batch in enumerate(train_loader):
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].to(device)
        labels = batch['labels'].to(device)
        if (step + 1) % args.accumulation_steps == 0:
            scheduler(step)
            optimizer.zero_grad()
        outputs = model(inputs)
        # loss = loss_fn(outputs, labels, model) / args.accumulation_steps
        loss = loss_fn(outputs, labels) / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            xm.optimizer_step(optimizer)
        xm.master_print(f"Step [{step + 1}/{len(train_loader)}], Loss: {loss.item()}")

    # Evaluate on OOD dataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                inputs = batch["images"]
                labels = batch["labels"]
            else:
                inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    accuracy = xm.mesh_reduce(f"ood_subset_for_hp_accuracy", accuracy, np.mean)
    return -accuracy


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    PASSWORD = "robust-ft"
    IP_ADDRESS = "10.66.112.3"
    USER = "root"
    DATABASE_NAME = "optuna"
    storage_url = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{IP_ADDRESS}/{DATABASE_NAME}"
    study = optuna.create_study(storage=storage_url, study_name=args.save, direction="maximize", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, device, args), n_trials=100)
    xm.master_print(study.best_params)

def main(args):
    logger = setup_logging(args)
    logger.info(args)
    xmp.spawn(_mp_fn, args=(args,), nprocs=8, start_method='spawn')


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    os.environ["PYTHONPATH"] = "${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft_tpu/"
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    main(args)
    xm.master_print("Run time: {:.3f} s".format(time.time() - start_time))


