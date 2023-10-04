import copy
import gc
import json
import logging
import os
import time
from functools import partial

import numpy as np
import optuna
import torch

import src.losses as losses
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.finetune import finetune_final, inner_finetune
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel, set_seed

logger = logging.getLogger('main')

@torch.no_grad()
def evaluate_net(net, dataloader, dataset, args):
    net = net.cuda()
    net.eval()

    total_correct = 0
    total_samples = 0

    # Introducing variables to save predictions, labels, and metadata
    all_labels, all_preds, all_metadata = [], [], []

    batch_prep_times, feedforward_times = [], []
    for batch in dataloader:
        torch.cuda.synchronize()
        start = time.time()
        data = maybe_dictionarize(batch)
        x = data['images'].cuda()
        y = data['labels'].cuda()
        torch.cuda.synchronize()
        batch_prep_times.append(time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        outputs = net(x)
        torch.cuda.synchronize()
        feedforward_times.append(time.time() - start)

        if (y == -1).any():  # Handle unlabeled parts of the batch
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (y == -1)
            y[mask] = pseudo_labels[mask]

        predictions = outputs.argmax(dim=1)
        correct = (predictions == y).sum().item()
        total_correct += correct
        total_samples += y.size(0)

        # Save labels, predictions, and metadata
        all_labels.append(y.cpu().clone().detach())
        all_preds.append(outputs.cpu().clone().detach())
        if 'metadata' in data:
            all_metadata.extend(data['metadata'])
    
    print(f"    Time per batch prep: {np.mean(batch_prep_times):.3f} x {len(batch_prep_times)} = {sum(batch_prep_times):.3f}", flush=True)
    print(f"    Time per eval batch: {np.mean(feedforward_times):.3f} x {len(feedforward_times)} = {sum(feedforward_times):.3f}", flush=True)

    accuracy = (total_correct / total_samples) * 100

    # Concatenate all saved tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Calculate post loop metrics if available
    if hasattr(dataset, 'post_loop_metrics'):
        metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
        if 'acc' not in metrics:
            metrics['acc'] = accuracy
    else:
        metrics = {'acc': accuracy}

    torch.cuda.empty_cache()
    return metrics

def print_hparams(hparams):
    print("\nHyperparameters:")
    for key, value in hparams.items():
        if not "dataw" in key:
            print(f"{key}: {value}")

def save_hparams(hparams, args):
    save_file = os.path.join(args.save, 'hparams.json')
    os.makedirs(args.save, exist_ok=True)
    hparams["seed"] = int(hparams["seed"])
    with open(save_file, 'w') as f:
        json.dump(hparams, f)

class HyperparameterSpace:
    def __init__(self, model, loss_type, dataset_name, num_losses=None, num_datapoints=None):
        self.model = model
        self.loss_type = loss_type
        self.dataset_name = dataset_name
        self.num_losses = num_losses
        self.num_datapoints = num_datapoints
        if loss_type == "LayerwiseLoss":
            assert num_losses is not None

    def _base_space_learned_loss(self, trial, prefix):
        return {
            f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-7, 1e-3, log=True),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0),
            **{f"{prefix}lossw_{i}": trial.suggest_float(f"{prefix}lossw_{i}", 1e-4, 10, log=True) for i in
               range(self.num_losses)}
        }

    def _base_space_layerwise_loss(self, trial, prefix):
        return {
            f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-7, 1e-3, log=True),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0),
            **{f"{prefix}lossw_{i}": trial.suggest_float(f"{prefix}lossw_{i}", 1e-5, 1e-2, log=True) for i in
               range(self.num_losses)}
        }

    def build_learned_loss_space(self, trial):
        hparams = self._base_space_learned_loss(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        if self.num_datapoints:
            hparams.update({f"dataw_{i}": trial.suggest_float(f"dataw_{i}", 0.0, 1.0) for i in range(self.num_datapoints)})
        return hparams

    def build_layerwise_loss_space(self, trial):
        hparams = {}
        layer_idx = 0
        self.model = extract_from_data_parallel(self.model)
        for name, module in self.model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    hparams.update(self._base_space_layerwise_loss(trial, f"{layer_idx}_"))
                    layer_idx += 1
                # Classification head of the model
                hparams.update(self._base_space_layerwise_loss(trial, f"{layer_idx}_"))
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
                        'lr': hyperparams[f"{layer_idx}_lr"],
                        'weight_decay': hyperparams[f"{layer_idx}_wd"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

        # Classification head of the model
        params_for_layer = {
            'params': model.classification_head.parameters(),
            'lr': hyperparams[f"{layer_idx}_lr"],
            'weight_decay': hyperparams[f"{layer_idx}_wd"]
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
            if int(k.split("_")[0]) == layer_idx:
                layer_loss_weights.append(hyperparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hyperparams[k]]
                layer_idx += 1
    else:
        assert num_losses is not None
        loss_weights = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(num_losses)])
    return loss_weights

def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader=None, image_encoder=None):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    for _ in range(args.repeats):
        start_time = time.time()
        current_net = copy.deepcopy(net)
        print(f"  Time to copy net: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        optimizer = create_optimizer(current_net, hyperparams, args.loss_type)
        initial_net_params = [p for p in net.parameters()]
        loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)
        print(f"  Time to construct loss: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        if args.pointwise_loss and "dataw_0" in hyperparams.keys():
            data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
            sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
            id_dataloader = get_dataloader(id_dataloader.dataset, is_train=True, args=args, sampler=sampler, image_encoder=None)
        print(f"  Time for data prep: {time.time() - start_time:.3f}", flush=True)

        set_seed(hyperparams["seed"])
        start_time = time.time()
        current_net = inner_finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key,
                                     unlabeled_dataloader=unlabeled_dataloader, image_encoder=image_encoder)
        print(f"  Time to finetune: {time.time() - start_time:.3f}", flush=True)
        set_seed(args.seed)

        val_results = dict()
        start_time = time.time()
        metrics = evaluate_net(current_net, ood_hp_dataloader, ood_hp_dataset, args=args)  # Changed this line to use the new evaluate_net
        print(f"  Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        if "IWildCam" in args.id:
            val_results[f"ood_subset_for_hp_accuracy"] = metrics['F1-macro_all']
            # val_results[f"ood_subset_for_hp_accuracy"] = metrics['acc']
        elif "FMOW" in args.id:
            val_results[f"ood_subset_for_hp_accuracy"] = metrics['acc_worst_region']
        else:
            val_results[f"ood_subset_for_hp_accuracy"] = metrics['acc']  # Using 'acc' from the metrics dict
        all_val_results.append(val_results)

    return all_val_results


def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    torch.cuda.empty_cache()

def auto_ft(args, model, id_dataloader, ood_hp_dataloader, ood_hp_dataset, max_evals, input_key, unlabeled_dataloader=None, image_encoder=None):
    """Automated fine-tuning process using Optuna."""
    def hp_objective_fn(trial, hspace):
        full_loop_start_time = time.time()
        start = time.time()
        hparams = hspace.build_space(trial)
        print(f"  Time to get hparams: {time.time() - start:.3f}", flush=True)

        start = time.time()
        _net = copy.deepcopy(model).cuda()
        print(f"  Time to copy model: {time.time() - start:.3f}", flush=True)

        val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader, image_encoder)
        print(f"Total time for autoft itreration: {time.time() - full_loop_start_time:.3f}", flush=True)
        return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        partial_hp_objective_fn = partial(hp_objective_fn, hspace=HyperparameterSpace(model, args.loss_type, args.id, args.num_losses, None))
        if args.pointwise_loss:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(partial_hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
        best_hparams = study.best_params

    breakpoint()
    loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
    initial_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weights, initial_params)
    optimizer = create_optimizer(model, best_hparams, args.loss_type)
    print_hparams(best_hparams)
    save_hparams(best_hparams, args)
    set_seed(best_hparams["seed"])
    ft_model = finetune_final(args, model, loss_fn, optimizer, id_dataloader, input_key, 100, unlabeled_dataloader, image_encoder)

    return ft_model