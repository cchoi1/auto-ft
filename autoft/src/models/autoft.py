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
    def __init__(self, model, losses, dataset_name, orig_lr, layerwise_loss=False, layerwise_opt=False, learn_lr_wd=True, relative_to_flyp=False):
        self.model = model
        self.losses = sorted(losses)
        self.dataset_name = dataset_name
        self.orig_lr = orig_lr
        self.layerwise_loss = layerwise_loss
        self.layerwise_opt = layerwise_opt
        self.learn_lr_wd = learn_lr_wd
        self.relative_to_flyp = relative_to_flyp

    def _base_loss_weight_space(self, trial, prefix):
        base_loss_weight_space = {}
        if "ce" in self.losses and not self.relative_to_flyp:
            base_loss_weight_space[f"{prefix}lossw_ce"] = 1.0
            for loss_type in self.losses:
                if loss_type in ["dcm", "entropy", "hinge", "flyp"]:
                    base_loss_weight_space[f"{prefix}lossw_{loss_type}"] = trial.suggest_float(
                        f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
        elif "flyp" in self.losses and self.relative_to_flyp:
            base_loss_weight_space[f"{prefix}lossw_flyp"] = 1.0
            for loss_type in self.losses:
                if loss_type in ["ce", "dcm", "entropy", "hinge"]:
                    base_loss_weight_space[f"{prefix}lossw_{loss_type}"] = trial.suggest_float(
                        f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
        return base_loss_weight_space

    def _base_norm_space(self, trial, prefix):
        return {
            **{f"{prefix}lossw_{loss_type}": trial.suggest_float(f"{prefix}lossw_{loss_type}", 1e-4, 10, log=True)
               for loss_type in self.losses if loss_type in ["l1init", "l1zero", "l2zero", "l2init"]},
        }

    def _base_lr_wd_space(self, trial, prefix):
        lr_lower_bound = 1e-2 * self.orig_lr
        lr_upper_bound = 1e2 * self.orig_lr
        if self.layerwise_opt:
            return {
                f"{prefix}lr": trial.suggest_float(f"{prefix}lr", lr_lower_bound, lr_upper_bound, log=True),
                f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0)
            }
        else:
            if "Flowers" in self.dataset_name:
                return {
                    f"{prefix}lr": trial.suggest_float(f"{prefix}lr", lr_lower_bound, lr_upper_bound, log=True),
                    f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0)
                }
            elif "PatchCamelyon" in self.dataset_name or "StanfordCars" in self.dataset_name or "IWildCam" in self.dataset_name:
                return {
                    f"{prefix}lr": trial.suggest_float(f"{prefix}lr", lr_lower_bound, lr_upper_bound, log=True),
                    f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 0.3)
                }
            else:
                return {
                    f"{prefix}lr": trial.suggest_float(f"{prefix}lr", lr_lower_bound, lr_upper_bound, log=True),
                    f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 1e-2, 0.3, log=True)
                }

    def build_space(self, trial):
        # Global hyperparameters: loss weights, seed, batch size
        hparams = self._base_loss_weight_space(trial, "")
        hparams["seed"] = trial.suggest_int("seed", 0, 10)
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
            if self.learn_lr_wd:
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
        if "lr" not in hparams.keys() and "wd" not in hparams.keys():
            hparams["lr"] = 1e-5
            hparams["wd"] = 0.2
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


def evaluate_hparams(args, net, hparams, dataloaders, ood_hp_dataset, input_key, fs_id_dataset=None, fs_val_dataset=None):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    current_net = copy.deepcopy(net)
    optimizer = create_optimizer(model=current_net, hparams=hparams, layerwise=args.layerwise_opt)
    initial_params = [p for p in net.parameters()]
    loss_weights = get_loss_weights(hparams=hparams, layerwise=args.layerwise_loss)
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)

    set_seed(hparams["seed"])
    current_net, all_metrics = inner_finetune(args, current_net, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset, fs_id_dataset, fs_val_dataset)
    set_seed(args.seed)
    if "IWildCam" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['F1-macro_all']
    elif "FMOW" in args.id:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['acc_worst_region']
    elif "sst2" in args.id:
        all_metrics[f"meta_learning_objective"] = -all_metrics["meta_objective"]['xent']
    elif "PatchCamelyon" in args.id and args.k is not None:
        all_metrics[f"meta_learning_objective"] = -all_metrics["meta_objective"]['xent']
    else:
        all_metrics[f"meta_learning_objective"] = all_metrics["meta_objective"]['acc']

    if args.xent_meta_objective:
        all_metrics[f"meta_learning_objective"] = -all_metrics["meta_objective"]['xent']

    return all_metrics


def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    torch.cuda.empty_cache()


def auto_ft_iteration(args, model, dataloaders, ood_hp_dataset, max_evals, input_key, fs_id_dataset=None, fs_val_dataset=None):
    """Automated fine-tuning process using Optuna."""
    def hp_objective_fn(trial, hspace):
        full_loop_start_time = time.time()
        hparams = hspace.build_space(trial)
        _net = copy.deepcopy(model).cuda()
        val_results = [
            evaluate_hparams(args, _net, hparams, dataloaders, ood_hp_dataset, input_key, fs_id_dataset, fs_val_dataset)
        ]
        print(f" Total time for autoft iteration: {time.time() - full_loop_start_time:.3f}")

        trial_file = os.path.join("logs", args.save, f"trial_{trial.number}.json")
        os.makedirs(args.save, exist_ok=True)
        with open(trial_file, 'w') as f:
            first_result = copy.deepcopy(val_results[0])
            first_result["hparams"] = hparams
            json.dump(first_result, f)
        # Maximize performance metric, prioritizing later trials in case of a tie
        obj = -np.mean([r["meta_learning_objective"] for r in val_results]) - trial.number * 1e-10
        return obj

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        if args.losses == ["flyp"]:
            best_hparams = {"seed": 0}
            best_hparams["lossw_flyp"] = 1.0
            best_hparams["lr"] = args.lr
            best_hparams["wd"] = args.wd
        elif args.losses == ["ce", "flyp"]:
            best_hparams = {"seed": 0}
            best_hparams["lossw_ce"] = 1.0
            best_hparams["lossw_flyp"] = 1.0
            best_hparams["lr"] = args.lr
            best_hparams["wd"] = args.wd
        else:
            if args.no_lr_wd:
                learn_lr_wd = False
            else:
                learn_lr_wd = True
            hspace = HyperparameterSpace(model=model, losses=args.losses, dataset_name=args.id, orig_lr=args.lr,
                                        layerwise_loss=args.layerwise_loss, layerwise_opt=args.layerwise_opt,
                                        learn_lr_wd=learn_lr_wd, relative_to_flyp=args.relative_to_flyp)
            partial_hp_objective_fn = partial(hp_objective_fn, hspace=hspace)

            pruner_kwargs = {'n_startup_trials': 5, 'n_warmup_steps': 30, 'interval_steps': 10}
            pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
            if args.optuna_sampler == "random":
                print(f"Using random sampler for optuna")
                study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=args.seed), direction="minimize", pruner=pruner)
            else:
                print(f"Using default TPE sampler for optuna, option={args.optuna_sampler}")
                study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(partial_hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
            best_hparams = study.best_params

    if "ce" in args.losses and "lossw_ce" not in best_hparams.keys():
        best_hparams["lossw_ce"] = 1.0
    if "flyp" in args.losses and "lossw_flyp" not in best_hparams.keys():
        best_hparams["lossw_flyp"] = 1.0
    loss_weights = get_loss_weights(hparams=best_hparams, layerwise=args.layerwise_loss)
    initial_params = [p for p in model.parameters()]
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)
    optimizer = create_optimizer(model=model, hparams=best_hparams, layerwise=args.layerwise_opt)
    print_hparams(hparams=best_hparams)
    save_hparams(hparams=best_hparams, args=args)
    set_seed(seed=best_hparams["seed"])
    if args.k is None:
        print_every = 100
    else:
        print_every = 1

    ft_model, val_metric = finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every, fs_id_dataset, fs_val_dataset)

    return {"model": ft_model, "val_metric": val_metric, "hparams": best_hparams}

def auto_ft(args, _model, dataloaders, ood_hp_dataset, max_evals, input_key, fs_id_dataset=None, fs_val_dataset=None):
    best_val_metric = np.inf
    best_hparams = None
    best_model = None
    best_iter = None
    for i in range(args.autoft_repeats):
        seed = np.random.randint(0, 1000)
        set_seed(seed)
        model = copy.deepcopy(_model)
        results = auto_ft_iteration(args, model, dataloaders, ood_hp_dataset, max_evals, input_key, fs_id_dataset, fs_val_dataset)
        print(f"\nIteration {i} has val loss {results['val_metric']}")
        if results["val_metric"] < best_val_metric:
            best_val_metric = results["val_metric"]
            best_hparams = copy.deepcopy(results["hparams"])
            best_model = copy.deepcopy(results["model"])
            best_iter = i
    print(f"\n\n-------Best Hyperparameters out of {args.autoft_repeats} runs-------\n "
          f"Iteration {best_iter}. {best_hparams} with Val Metric {best_val_metric}")

    return best_model, best_val_metric