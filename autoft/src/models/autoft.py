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
from src.datasets.common import get_dataloader
from src.models.finetune import inner_finetune, finetune_final
from src.models.train_utils import evaluate_hp, print_hparams, set_seed, create_layerwise_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('main')
devices = list(range(torch.cuda.device_count()))

def build_hparams_space_learnedloss(trial, num_losses, num_datapoints=None):
    """Build the hyperparameter search space for Optuna."""
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    wd = trial.suggest_float("wd", 0.0, 1.0)
    loss_weights = [trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2) for i in range(num_losses)]
    hparams = {"lr": lr, "wd": wd, "seed": trial.suggest_int("seed", 0, 100)}
    for i, loss_weight in enumerate(loss_weights):
        hparams[f"lossw_{i}"] = loss_weight
    if num_datapoints is not None:
        hparams.update({f"dataw_{i}": trial.suggest_float(f"dataw_{i}", 0.0, 1.0) for i in range(num_datapoints)})
    return hparams


def build_hparams_space_layerwiseloss(trial, num_losses, model, num_datapoints=None):
    """Build the hyperparameter search space for Optuna."""
    hparams = {}

    # Embedding and transformer block hyperparameters
    if isinstance(model, torch.nn.DataParallel):
        model = [c for c in model.children()][0]  # Extract from nn.DataParallel
    layer_idx = 0
    for name, module in model.image_encoder.named_children():
        if name == 'model':
            # Initial Convolutional layer
            hparams[f"lr_{layer_idx}"] = trial.suggest_loguniform(f"lr_{layer_idx}", 1e-5, 1e-2)
            hparams[f"wd_{layer_idx}"] = trial.suggest_float(f"wd_{layer_idx}", 0.0, 1.0)
            for i in range(num_losses):
                hparams[f"lossw_{layer_idx}_{i}"] = trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2)
            layer_idx += 1

            # Layer normalization before the transformer
            hparams[f"lr_{layer_idx}"] = trial.suggest_loguniform(f"lr_{layer_idx}", 1e-5, 1e-2)
            hparams[f"wd_{layer_idx}"] = trial.suggest_float(f"wd_{layer_idx}", 0.0, 1.0)
            for i in range(num_losses):
                hparams[f"lossw_{layer_idx}_{i}"] = trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2)
            layer_idx += 1

            # Transformer blocks
            for block in module.visual.transformer.resblocks:
                for layer in block.children():
                    hparams[f"lr_{layer_idx}"] = trial.suggest_loguniform(f"lr_{layer_idx}", 1e-5, 1e-2)
                    hparams[f"wd_{layer_idx}"] = trial.suggest_float(f"wd_{layer_idx}", 0.0, 1.0)
                    for i in range(num_losses):
                        hparams[f"lossw_{layer_idx}_{i}"] = trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2)
                    layer_idx += 1

    # Classification head of the model
    hparams[f"lr_{layer_idx}"] = trial.suggest_loguniform(f"lr_{layer_idx}", 1e-5, 1e-2)
    hparams[f"wd_{layer_idx}"] = trial.suggest_float(f"wd_{layer_idx}", 0.0, 1.0)
    for i in range(num_losses):
        hparams[f"lossw_{layer_idx}_{i}"] = trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2)
    layer_idx += 1

    hparams["seed"] = trial.suggest_int("seed", 0, 100)

    return hparams


def build_hparams_space(trial, num_losses, model=None, num_datapoints=None):
    if model is not None:
        return build_hparams_space_layerwiseloss(trial, num_losses, model, num_datapoints)
    else:
        return build_hparams_space_learnedloss(trial, num_losses, num_datapoints)


def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    for _ in range(args.repeats):
        start_time = time.time()
        if args.distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            current_net = DDP(copy.deepcopy(net).to(local_rank), device_ids=[local_rank])
        else:
            current_net = copy.deepcopy(net).cuda()
        print(f"Time to copy net: {time.time() - start_time:.3f}", flush=True)

        set_seed(hyperparams["seed"])
        if args.loss_type == "LayerwiseLoss":
            optimizer = create_layerwise_optimizer(current_net, hyperparams)
        else:
            optimizer = torch.optim.AdamW(current_net.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
        if args.loss_type == "LayerwiseLoss":
            loss_weight_keys = [k for k in hyperparams.keys() if "lossw" in k]
            layer_idx = 0
            layer_loss_weights = []
            loss_weight_hparams = []
            for k in loss_weight_keys:
                if int(k.split("_")[1]) == layer_idx:
                    layer_loss_weights.append(hyperparams[k])
                else:
                    loss_weight_hparams.append(torch.tensor(layer_loss_weights))
                    layer_loss_weights = [hyperparams[k]]
                    layer_idx += 1
        else:
            loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(args.num_losses)])
        start_time = time.time()
        initial_net_params = [p for p in net.parameters()]
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)
        print(f"Time to create loss fn: {time.time() - start_time:.3f}", flush=True)
        start_time = time.time()
        if args.pointwise_loss and "dataw_0" in hyperparams.keys():
            data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
            sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
            id_dataloader = get_dataloader(id_dataloader.dataset, is_train=True, args=args, sampler=sampler, image_encoder=None)
        current_net = inner_finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        print(f"Time to finetune: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        val_results = dict()
        loss, accuracy = evaluate_hp(current_net, ood_hp_dataloader)
        print(f"Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        val_results[f"ood_subset_for_hp_loss"] = loss
        val_results[f"ood_subset_for_hp_accuracy"] = accuracy
        all_val_results.append(val_results)

    return all_val_results

def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals, input_key, print_every=None):
    """Automated fine-tuning process using Optuna."""
    def hp_objective_fn(trial):
        num_datapoints = len(id_dataloader.dataset) if args.pointwise_loss else None
        model_copy = copy.deepcopy(model) if args.loss_type == "LayerwiseLoss" else None
        hparams = build_hparams_space(trial, args.num_losses, model_copy, num_datapoints=num_datapoints)
        if args.distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            _net = DDP(copy.deepcopy(model).to(local_rank), device_ids=[local_rank])
        else:
            _net = copy.deepcopy(model)
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

    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        _model = DDP(copy.deepcopy(model).to(local_rank), device_ids=[local_rank])
    else:
        _model = copy.deepcopy(model).cuda()
    set_seed(best_hparams["seed"])
    if args.loss_type == "LayerwiseLoss":
        loss_weight_keys = [k for k in best_hparams.keys() if f"lossw" in k]
        layer_idx = 0
        layer_loss_weights = []
        loss_weight_hparams = []
        for k in loss_weight_keys:
            if int(k.split("_")[1]) == layer_idx:
                layer_loss_weights.append(best_hparams[k])
            else:
                loss_weight_hparams.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [best_hparams[k]]
                layer_idx += 1
    else:
        loss_weight_hparams = torch.tensor([best_hparams[f"lossw_{i}"] for i in range(args.num_losses)])
    model_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, model_params)
    if args.loss_type == "LayerwiseLoss":
        optimizer = create_layerwise_optimizer(_model, best_hparams)
    else:
        optimizer = torch.optim.AdamW(_model.parameters(), lr=best_hparams["lr"], weight_decay=best_hparams["wd"])
    sampler = None
    if args.pointwise_loss:
        data_weights = torch.tensor([best_hparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
        sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
    print_every = 100 if args.plot else None
    _model = finetune_final(args, _model, loss_fn, optimizer, id_dataloader.dataset, input_key, print_every=print_every, sampler=sampler)
    print_hparams(best_hparams)

    # Saving hyperparams
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        filepath = os.path.join(args.save, "best_hparams.json")
        with open(filepath, 'w') as f:
            json.dump(best_hparams, f)
        print(f"Saved best_hparams to {filepath}")

    return best_hparams