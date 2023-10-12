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
from functools import partial

from src.losses import LearnedLoss
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.finetune import inner_finetune, finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel, set_seed

logger = logging.getLogger('main')

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
    def __init__(self, model, losses, dataset_name, layerwise_loss=False, layerwise_opt=False):
        self.model = model
        self.losses = losses
        self.dataset_name = dataset_name
        self.layerwise_loss = layerwise_loss
        self.layerwise_opt = layerwise_opt

    def _base_loss_weight_space(self, trial, prefix):
        return {
            **{f"{prefix}lossw_{loss_type}": trial.suggest_float(f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
               for loss_type in self.losses if loss_type in ["ce", "hinge", "entropy", "dcm", "flyp"]},
        }

    def _base_norm_space(self, trial, prefix):
        return {
            **{f"{prefix}lossw_{loss_type}": trial.suggest_float(f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
               for loss_type in self.losses if loss_type in ["l1zero", "l2zero", "l1init", "l2init"]},
        }

    def _base_lr_wd_space(self, trial, prefix):
        return {
            f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-7, 1e-3, log=True),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0)
        }

    def build_space(self, trial):
        # Global hyperparameters: loss weights and random seed
        hparams = self._base_loss_weight_space(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        if self.layerwise_loss: # per-layer LRs, WDs, L1/L2 norms
            layer_idx = 0
            self.model = extract_from_data_parallel(self.model)
            for name, module in self.model.image_encoder.named_children():
                if name == 'model':
                    for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                        hparams.update(self._base_norm_space(trial, f"{layer_idx}_"))
                        layer_idx += 1
                    # Classification head of the model
                    hparams.update(self._base_norm_space(trial, f"{layer_idx}_"))
        else:
            hparams.update(self._base_norm_space(trial, ""))
        if self.layerwise_opt:
            layer_idx = 0
            self.model = extract_from_data_parallel(self.model)
            for name, module in self.model.image_encoder.named_children():
                if name == 'model':
                    for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                        hparams.update(self._base_lr_wd_space(trial, f"{layer_idx}_"))
                        layer_idx += 1
                    # Classification head of the model
                    hparams.update(self._base_lr_wd_space(trial, f"{layer_idx}_"))
        else:
            hparams.update(self._base_lr_wd_space(trial, ""))
        return hparams


def create_optimizer(model, hyperparams, layerwise=False):
    model = extract_from_data_parallel(model)
    assert isinstance(model, ImageClassifier), "Expected model to be an instance of ImageClassifier"
    if layerwise:
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


def get_loss_weights(hyperparams, losses, layerwise):
    global_loss_weight_keys = [k for k in hyperparams.keys() if "lossw" in k and "_lossw" not in k]
    global_loss_weights = torch.tensor([hyperparams[k] for k in global_loss_weight_keys])
    if layerwise:
        layerwise_loss_weight_keys = [k for k in hyperparams.keys() if "_lossw" in k]
        layerwise_loss_weight_keys = sorted(layerwise_loss_weight_keys, key=lambda x: int(x.split("_")[0]))
        layer_idx = 0
        layer_loss_weights = []
        loss_weights = []
        for k in layerwise_loss_weight_keys:
            if int(k.split("_")[0]) == layer_idx:
                layer_loss_weights.append(hyperparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hyperparams[k]]
                layer_idx += 1
        layerwise_loss_weights = torch.stack(loss_weights)
        global_loss_weights = global_loss_weights.expand(layerwise_loss_weights.shape[0], -1)
        loss_weights = torch.cat([global_loss_weights, layerwise_loss_weights], dim=1)
    else:
        loss_weights = global_loss_weights
    return loss_weights


def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader=None, image_encoder=None):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    start_time = time.time()
    current_net = copy.deepcopy(net)
    print(f"  Time to copy net: {time.time() - start_time:.3f}", flush=True)

    start_time = time.time()
    optimizer = create_optimizer(current_net, hyperparams, args.loss_type)
    initial_net_params = [p for p in net.parameters()]
    loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
    loss_fn = LearnedLoss(loss_weight_hparams, initial_net_params)
    print(f"  Time to construct loss: {time.time() - start_time:.3f}", flush=True)

    start_time = time.time()
    if args.pointwise_loss and "dataw_0" in hyperparams.keys():
        data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
        sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
        id_dataloader = get_dataloader(id_dataloader.dataset, is_train=True, args=args, sampler=sampler, image_encoder=None)
    print(f"  Time for data prep: {time.time() - start_time:.3f}", flush=True)

    set_seed(hyperparams["seed"])
    dataloaders = {"id": id_dataloader, "ood_hp": ood_hp_dataloader, "unlabeled": unlabeled_dataloader}
    current_net, all_metrics = inner_finetune(args, current_net, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset, image_encoder=image_encoder)
    set_seed(args.seed)
    if "IWildCam" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["last"]['F1-macro_all']
        # all_metrics[f"meta_learning_objective"] = all_metrics["last"]['acc']
    elif "FMOW" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["last"]['acc_worst_region']
    else:
        all_metrics[f"meta_learning_objective"] = all_metrics["last"]['acc']

    return all_metrics


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

        val_results = [evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader, image_encoder) for _ in range(args.repeats)]
        print(f"Total time for autoft iteration: {time.time() - full_loop_start_time:.3f}", flush=True)

        trial_file = os.path.join("logs", args.save, f"trial_{trial.number}.json")
        os.makedirs(args.save, exist_ok=True)
        with open(trial_file, 'w') as f:
            first_result = copy.deepcopy(val_results[0])
            first_result["hparams"] = hparams
            json.dump(first_result, f)
        return -np.mean([r["meta_learning_objective"] for r in val_results])  # maximize accuracy

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        partial_hp_objective_fn = partial(hp_objective_fn, hspace=HyperparameterSpace(model, args.loss_type, args.id, args.num_losses, None))
        if args.pointwise_loss:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if args.optuna_sampler == "random":
            print(f"Using random sampler for optuna")
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=args.seed), direction="minimize")
        else:
            print(f"Using default TPE sampler for optuna, option={args.optuna_sampler}")
            study = optuna.create_study(direction="minimize")
        study.optimize(partial_hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
        best_hparams = study.best_params

    loss_weights = get_loss_weights(best_hparams, args.losses, layerwise=args.layerwise_loss)
    initial_params = [p for p in model.parameters()]
    loss_fn = LearnedLoss(args.losses, loss_weights, initial_params)
    optimizer = create_optimizer(model, best_hparams, args.layerwise_opt)
    print_hparams(best_hparams)
    save_hparams(best_hparams, args)
    set_seed(best_hparams["seed"])
    ft_model = finetune_final(args, model, loss_fn, optimizer, id_dataloader, input_key, 100, unlabeled_dataloader)

    return ft_model