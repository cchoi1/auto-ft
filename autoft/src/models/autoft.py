import copy
import gc
import json
import logging
import os
import time

import numpy as np
import optuna
import src.losses as losses
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from src.datasets.common import get_dataloader
from src.models.finetune import inner_finetune, finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import is_tpu_available, get_device, extract_from_data_parallel, set_seed

logger = logging.getLogger('main')

@torch.no_grad()
def evaluate_net(net, dataloader):
    losses, predictions, targets = [], [], []
    device = get_device()

    if is_tpu_available():
        devices = xm.get_xla_supported_devices()
        net = xm.replicate(net, devices=devices)
        dataloader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
    elif torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch["images"].to(device)
            y = batch["labels"].to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
        outputs = net(x)
        if (y == -1).any(): # if parts of the batch are unlabeled, replace them with their pseudolabels
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (y == -1)
            y[mask] = pseudo_labels[mask]
        loss = F.cross_entropy(outputs, y)

        targets.append(y.cpu())
        losses.append(loss.cpu())
        predictions.append(outputs.argmax(dim=1).cpu())

        if is_tpu_available():
            xm.mark_step()

    if is_tpu_available():
        losses = xm.all_reduce("mean", torch.stack(losses)).mean()
        predictions = torch.cat(xm.all_gather(predictions, dim=0))
        targets = torch.cat(xm.all_gather(targets, dim=0))
    else:
        losses = torch.stack(losses).mean()
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

    accuracy = (predictions == targets).float().mean() * 100
    return loss.item(), accuracy.item()

def print_hparams(hparams):
    xm.master_print("\nHyperparameters:")
    for key, value in hparams.items():
        if not "dataw" in key:
            xm.master_print(f"{key}: {value:.4f}")

def save_hparams(hparams):
    should_save = xm.is_master_original() if is_tpu_available() else True
    if should_save:
        if not os.path.exists("hparams"):
            os.makedirs("hparams")
        with open(f"hparams/{hparams['trial_id']}.json", 'w') as f:
            json.dump(hparams, f)

class HyperparameterSpace:
    def __init__(self, model, loss_type, num_losses=None, num_datapoints=None):
        self.model = model
        self.loss_type = loss_type
        self.num_losses = num_losses
        self.num_datapoints = num_datapoints
        if loss_type == "LayerwiseLoss":
            assert num_losses is not None

    def _base_space(self, trial, prefix):
        return {
            f"{prefix}lr": trial.suggest_loguniform(f"{prefix}lr", 1e-5, 1e-2),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0),
            **{f"{prefix}lossw_{i}": trial.suggest_loguniform(f"{prefix}lossw_{i}", 1e-5, 1e-2) for i in range(self.num_losses)}
        }

    def build_learned_loss_space(self, trial):
        hparams = self._base_space(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        if self.num_datapoints:
            hparams.update({f"dataw_{i}": trial.suggest_float(f"dataw_{i}", 0.0, 1.0) for i in range(self.num_datapoints)})
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


def create_optimizer(model, hyperparams, loss_type):
    model = extract_from_data_parallel(model)
    assert isinstance(model, ImageClassifier), "Expected model to be an instance of ImageClassifier"
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

def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    for _ in range(args.repeats):
        current_net = copy.deepcopy(net)
        set_seed(hyperparams["seed"])
        optimizer = create_optimizer(current_net, hyperparams, args.loss_type)

        initial_net_params = [p for p in net.parameters()]
        loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)

        start_time = time.time()
        if args.pointwise_loss and "dataw_0" in hyperparams.keys():
            data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
            sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
            id_dataloader = get_dataloader(id_dataloader.dataset, is_train=True, args=args, sampler=sampler, image_encoder=None)
        current_net = inner_finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        print(f"Time to finetune: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        val_results = dict()
        loss, accuracy = evaluate_net(current_net, ood_hp_dataloader)
        print(f"Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        val_results[f"ood_subset_for_hp_loss"] = loss
        val_results[f"ood_subset_for_hp_accuracy"] = accuracy
        all_val_results.append(val_results)

    return all_val_results

def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    if is_tpu_available():
        xm.rendezvous("clear_memory")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals, input_key, print_every=None):
    """Automated fine-tuning process using Optuna."""
    def hp_objective_fn(trial):
        num_datapoints = len(id_dataloader.dataset) if args.pointwise_loss else None
        hparams = HyperparameterSpace(model, args.loss_type, args.num_losses, num_datapoints).build_space(trial)
        device = get_device()
        _net = copy.deepcopy(model).to(device)
        val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, input_key)
        return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        if args.pointwise_loss:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
        best_hparams = study.best_params

    device = get_device()
    ft_model = copy.deepcopy(model).to(device)
    set_seed(best_hparams["seed"])
    loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
    model_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weights, model_params)
    optimizer = create_optimizer(ft_model, best_hparams, args.loss_type)
    sampler = None
    if args.pointwise_loss:
        data_weights = torch.tensor([best_hparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
        sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
    ft_model = finetune_final(args, ft_model, loss_fn, optimizer, id_dataloader.dataset, input_key, print_every=print_every, sampler=sampler)
    print_hparams(best_hparams)
    save_hparams(best_hparams)

    return best_hparams