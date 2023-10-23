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

from src.datasets.common import get_autoft_dataloaders
from src.losses import LearnedLoss
from src.models.finetune import inner_finetune, finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel, set_seed, print_hparams, save_hparams

logger = logging.getLogger('main')

class HyperparameterSpace:
    def __init__(self, model, losses, dataset_name, orig_lr, layerwise_loss=False, layerwise_opt=False, learn_batch_size=False):
        self.model = model
        self.losses = losses
        self.dataset_name = dataset_name
        self.orig_lr = orig_lr
        self.layerwise_loss = layerwise_loss
        self.layerwise_opt = layerwise_opt
        self.learn_batch_size = learn_batch_size

    def _base_loss_weight_space(self, trial, prefix):
        return {
            **{f"{prefix}lossw_{loss_type}": trial.suggest_float(f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
               for loss_type in self.losses if loss_type in ["hinge", "entropy", "dcm", "flyp"]},
            f"{prefix}lossw_ce": 1.0    # set cross-entropy loss weight to 1.0
        }

    def _base_norm_space(self, trial, prefix):
        return {
            **{f"{prefix}lossw_{loss_type}": trial.suggest_float(f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
               for loss_type in self.losses if loss_type in ["l1zero", "l2zero", "l1init", "l2init"]},
        }

    def _base_lr_wd_space(self, trial, prefix):
        lr_lower_bound = 1e-2 * self.orig_lr
        lr_upper_bound = 1e2 * self.orig_lr
        return {
            f"{prefix}lr": trial.suggest_float(f"{prefix}lr", lr_lower_bound, lr_upper_bound, log=True),
            f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0)
        }

    def build_space(self, trial):
        # Global hyperparameters: loss weights, seed, batch size
        hparams = self._base_loss_weight_space(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        if self.learn_batch_size:
            hparams["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
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


def create_optimizer(model, hparams, layerwise=False):
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
                        'lr': hparams[f"{layer_idx}_lr"],
                        'weight_decay': hparams[f"{layer_idx}_wd"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

        # Classification head of the model
        params_for_layer = {
            'params': model.classification_head.parameters(),
            'lr': hparams[f"{layer_idx}_lr"],
            'weight_decay': hparams[f"{layer_idx}_wd"]
        }
        layerwise_params.append(params_for_layer)
        optimizer = torch.optim.AdamW(layerwise_params)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])
    return optimizer


def get_loss_weights(hparams, layerwise):
    global_loss_weight_keys = [k for k in sorted(hparams.keys()) if "lossw" in k and "_lossw" not in k]
    global_loss_weights = torch.tensor([hparams[k] for k in global_loss_weight_keys])
    if layerwise:
        layerwise_loss_weight_keys = [k for k in hparams.keys() if "_lossw" in k]
        layerwise_loss_weight_keys = sorted(layerwise_loss_weight_keys,
                                            key=lambda x: (int(x.split("_")[0]), x.split("_")[2]))
        layer_idx = 0
        layer_loss_weights = []
        loss_weights = []
        for k in layerwise_loss_weight_keys:
            if int(k.split("_")[0]) == layer_idx:
                layer_loss_weights.append(hparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hparams[k]]
                layer_idx += 1
        layerwise_loss_weights = torch.stack(loss_weights)
        global_loss_weights = global_loss_weights.expand(layerwise_loss_weights.shape[0], -1)
        loss_weights = torch.cat([global_loss_weights, layerwise_loss_weights], dim=1)
    else:
        loss_weights = global_loss_weights
    return loss_weights


def evaluate_hparams(args, net, hparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader=None):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    start_time = time.time()
    current_net = copy.deepcopy(net)
    print(f"  Time to copy net: {time.time() - start_time:.3f}")

    start_time = time.time()
    optimizer = create_optimizer(model=current_net, hparams=hparams, layerwise=args.layerwise_opt)
    initial_params = [p for p in net.parameters()]
    loss_weights = get_loss_weights(hparams=hparams, layerwise=args.layerwise_loss)
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)
    print(f"  Time to construct loss: {time.time() - start_time:.3f}")

    set_seed(hparams["seed"])
    dataloaders = {"id": id_dataloader, "ood_hp": ood_hp_dataloader, "unlabeled": unlabeled_dataloader}
    if "batch_size" in hparams.keys():
        start_time = time.time()
        all_datasets = {"id": id_dataloader.dataset, "ood_subset_for_hp": ood_hp_dataloader.dataset}
        if unlabeled_dataloader is not None:
            all_datasets["unlabeled"] = unlabeled_dataloader.dataset
        args.batch_size = hparams["batch_size"]
        dataloaders = get_autoft_dataloaders(args=args, model=current_net, all_datasets=all_datasets)
        print(f"  Time to get re-construct dataloaders: {time.time() - start_time:.3f}")
    current_net, all_metrics = inner_finetune(args, current_net, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset)
    set_seed(args.seed)
    if "IWildCam" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['F1-macro_all']
    elif "FMOW" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['acc_worst_region']
    else:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['acc']

    return all_metrics


def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    torch.cuda.empty_cache()


def auto_ft(args, model, id_dataloader, ood_hp_dataloader, ood_hp_dataset, max_evals, input_key, unlabeled_dataloader=None):
    """Automated fine-tuning process using Optuna."""
    def hp_objective_fn(trial, hspace):
        full_loop_start_time = time.time()
        start = time.time()
        hparams = hspace.build_space(trial)
        print(f"  Time to get hparams: {time.time() - start:.3f}")

        start = time.time()
        _net = copy.deepcopy(model).cuda()
        print(f"  Time to copy model: {time.time() - start:.3f}")

        val_results = [evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, ood_hp_dataset, input_key, unlabeled_dataloader) for _ in range(args.repeats)]
        print(f"    Total time for autoft iteration: {time.time() - full_loop_start_time:.3f}")

        trial_file = os.path.join("logs", args.save, f"trial_{trial.number}.json")
        os.makedirs(args.save, exist_ok=True)
        with open(trial_file, 'w') as f:
            first_result = copy.deepcopy(val_results[0])
            first_result["hparams"] = hparams
            json.dump(first_result, f)
        return -np.mean([r["meta_learning_objective"] for r in val_results])    # maximize performance metric

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        hspace = HyperparameterSpace(model=model, losses=args.losses, dataset_name=args.id, orig_lr=args.lr,
                                     layerwise_loss=args.layerwise_loss, layerwise_opt=args.layerwise_opt,
                                     learn_batch_size=args.learn_batch_size)
        partial_hp_objective_fn = partial(hp_objective_fn, hspace=hspace)

        if args.optuna_sampler == "random":
            print(f"Using random sampler for optuna")
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=args.seed), direction="minimize")
        else:
            print(f"Using default TPE sampler for optuna, option={args.optuna_sampler}")
            study = optuna.create_study(direction="minimize")
        study.optimize(partial_hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
        best_hparams = study.best_params

    if "ce" in args.losses and "lossw_ce" not in best_hparams.keys():
        best_hparams["lossw_ce"] = 1.0
    loss_weights = get_loss_weights(hparams=best_hparams, layerwise=args.layerwise_loss)
    initial_params = [p for p in model.parameters()]
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)
    optimizer = create_optimizer(model=model, hparams=best_hparams, layerwise=args.layerwise_opt)
    print_hparams(hparams=best_hparams)
    save_hparams(hparams=best_hparams, args=args)
    set_seed(seed=best_hparams["seed"])
    if "batch_size" in best_hparams.keys():
        args.batch_size = best_hparams["batch_size"]
        all_datasets = {"id": id_dataloader.dataset, "ood_subset_for_hp": ood_hp_dataloader.dataset}
        if unlabeled_dataloader is not None:
            all_datasets["unlabeled"] = unlabeled_dataloader.dataset
        dataloaders = get_autoft_dataloaders(args=args, model=model, all_datasets=all_datasets)
        id_dataloader = dataloaders["id"]
        unlabeled_dataloader = dataloaders["unlabeled"]
    ft_model = finetune_final(args, model, loss_fn, optimizer, id_dataloader, input_key, 100, unlabeled_dataloader)

    return ft_model