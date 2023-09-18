import gc
import json
import logging
import os
from contextlib import contextmanager
from functools import partial

import numpy as np
import optuna
import src.losses as losses
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.finetune import finetune
from src.models.modeling import ImageClassifier
from src.models.utils import set_seed

logger = logging.getLogger('main')

@torch.no_grad()
def evaluate_net(net, dataloader):
    net.eval()
    predictions, targets = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch["images"]
            y = batch["labels"]
        else:
            x, y = batch
        outputs = net(x)
        predictions.append(outputs.argmax(dim=1).cpu().detach().numpy())
        targets.append(y.cpu().detach().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    accuracy = (predictions == targets).mean() * 100
    return accuracy


def print_hparams(hparams):
    """Print hyperparameters, excluding dataweights."""
    xm.master_print("\nHyperparameters:")
    for key, value in hparams.items():
        if not "dataw" in key:
            xm.master_print(f"{key}: {value:.4f}")


def save_hparams(args, hparams):
    """Save hyperparameters to a file."""
    if xm.is_master_ordinal():
        os.makedirs(args.save, exist_ok=True)
        hparams_path = os.path.join(args.save, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f)


def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    """Clear TPU memory in between hyperparameter optimization trials."""
    gc.collect()


class HyperparameterSpace:
    def __init__(self, model, loss_type, num_losses=None):
        self.model = model
        self.loss_type = loss_type
        self.num_losses = num_losses
        if loss_type == "LayerwiseLoss":
            assert num_losses is not None

    def _base_space(self, trial, prefix, use_none=False):
        if use_none:
            return {
                f"{prefix}lr": None,
                f"{prefix}wd": None,
                **{f"{prefix}lossw_{i}": None for i in range(self.num_losses)}
            }
        else:
            return {
                f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-8, 1e-2, log=True),
                f"{prefix}wd": trial.suggest_float(f"{prefix}wd", 0.0, 1.0),
                **{f"{prefix}lossw_{i}": trial.suggest_float(f"{prefix}lossw_{i}", 1e-5, 1e-2, log=True) for i in range(self.num_losses)}
            }

    def build_learned_loss_space(self, trial, use_none=False):
        hparams = self._base_space(trial, "", use_none)
        if use_none:
            hparams["seed"] = None
        else:
            hparams["seed"] = trial.suggest_int("seed", 0, 100)
        return hparams

    def build_layerwise_loss_space(self, trial, use_none=False):
        #TODO add use_none for layerwise loss
        hparams = {}
        layer_idx = 0
        for name, module in self.model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    hparams.update(self._base_space(trial, f"{layer_idx}_", use_none))
                    layer_idx += 1
                # Classification head of the model
                hparams.update(self._base_space(trial, f"{layer_idx}_", use_none))
        hparams["seed"] = trial.suggest_int("seed", 0, 100)
        return hparams

    def build_space(self, trial, use_none=False):
        if self.loss_type == "LayerwiseLoss":
            return self.build_layerwise_loss_space(trial, use_none)
        else:
            assert self.num_losses is not None
            return self.build_learned_loss_space(trial, use_none)
        del self.model
        gc.collect()


def create_optimizer(model, hyperparams, loss_type):
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


@contextmanager
def managed_storage(storage_url):
    storage = optuna.storages.get_storage(storage_url)
    try:
        yield storage
    finally:
        storage._backend.engine.dispose()


def get_or_create_study(study_name, storage_url, direction="minimize", resume_study=False):
    """
    Load a study if it exists, otherwise create a new one.

    Parameters:
    - study_name (str): Name of the study.
    - storage_url (str): Storage URL for the database.
    - direction (str): Optimization direction ("minimize" or "maximize").

    Returns:
    - study: An Optuna study object.
    """
    with managed_storage(storage_url) as storage:
        all_study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        pruner = optuna.pruners.MedianPruner()
        existing_study_names = [summary.study_name for summary in all_study_summaries]
        if study_name in existing_study_names:
            if not resume_study:
                msg = f"Found existing study {study_name}. Deleting it and creating a new one."
                xm.master_print(msg)
                logger.info(msg)
                optuna.delete_study(study_name=study_name, storage=storage_url)
                study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction, pruner=pruner)
            else:
                msg = f"Found existing study {study_name}. Loading it."
                xm.master_print(msg)
                logger.info(msg)
                study = optuna.load_study(study_name=study_name, storage=storage_url, pruner=pruner)
        else:
            msg = f"Could not find existing study {study_name}. Creating a new one."
            xm.master_print(msg)
            logger.info(msg)
            study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction, pruner=pruner)

    return study


def compute_loss(loss_fn, inputs, labels, model):
    """Computes the loss using either LearnedLoss, LayerwiseLoss, or default method."""
    outputs = model(inputs)
    if isinstance(loss_fn, (LearnedLoss, LayerwiseLoss)):
        return loss_fn(outputs, labels, model)
    return loss_fn(outputs, labels)

# @profile
def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    net_weights = net.state_dict()
    initial_net_params = [p for p in net.parameters()]
    optimizer = create_optimizer(net, hyperparams, args.loss_type)
    loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
    for _ in range(args.repeats):
        net.load_state_dict(net_weights)
        set_seed(hyperparams["seed"])
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)

        # net = finetune_helper(args, net, loss_fn, optimizer, id_dataloader, input_key, steps=args.inner_steps, accumulation_steps=args.accumulation_steps)
        net.train()

        effective_step = 0
        for step, batch in enumerate(id_dataloader):
            if step >= args.inner_steps:
                break
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key]
            labels = batch['labels']
            if (step + 1) % args.accumulation_steps == 0:
                effective_step += 1
                optimizer.zero_grad()
            loss = compute_loss(loss_fn, inputs, labels, net) / args.accumulation_steps
            loss.backward()
            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                xm.optimizer_step(optimizer)
            step += 1

        with torch.no_grad():
            accuracy = evaluate_net(net, ood_hp_dataloader)
        val_results = {"ood_subset_for_hp_loss": loss, "ood_subset_for_hp_acc": accuracy}
        all_val_results.append(val_results)
        del loss_fn, loss, accuracy

    del net_weights, net, optimizer, initial_net_params, loss_weight_hparams, id_dataloader, ood_hp_dataloader
    gc.collect()
    return all_val_results

@contextmanager
def optuna_study_context(study_name, storage_url):
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    try:
        yield study
    finally:
        if hasattr(study._storage, "_backend") and hasattr(study._storage._backend, "engine"):
            study._storage._backend.engine.dispose()


def hp_objective_fn(trial, args, model, id_dataloader, ood_hp_dataloader, input_key):
    hspace = HyperparameterSpace(model, args.loss_type, args.num_losses)
    hparams = hspace.build_space(trial)
    del hspace
    val_results = evaluate_hparams(args, model, hparams, id_dataloader, ood_hp_dataloader, input_key)
    acc_on_this_core = torch.tensor(np.mean([r["ood_subset_for_hp_acc"] for r in val_results])).to(xm.xla_device())
    summed_acc = xm.all_reduce("sum", acc_on_this_core)
    avg_acc = summed_acc / xm.xrt_world_size()
    del hparams, val_results
    gc.collect()
    return -avg_acc


def _mp_auto_ft(rank, args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key):
    """Automated fine-tuning using Optuna."""
    torch.manual_seed(args.seed + rank)
    xm.master_print("Starting _mp_auto_ft")
    id_dataloader = get_dataloader(id_dataset, is_train=True, args=args, image_encoder=None)
    ood_hp_dataloader = get_dataloader(ood_hp_dataset, is_train=True, args=args, image_encoder=None)
    device = xm.xla_device()
    model = model.to(device)

    objective = partial(hp_objective_fn, args=args, model=model, id_dataloader=id_dataloader, ood_hp_dataloader=ood_hp_dataloader, input_key=input_key)
    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        with optuna_study_context(args.save, storage_url) as study:
            xm.master_print("Loaded study")
            study.optimize(objective, n_trials=max_evals, timeout=None, gc_after_trial=True)
            best_trial = study.best_trial
            xm.master_print(f"\nBest trial {best_trial.number} with value {best_trial.value}")
            best_hparams = best_trial.params
    if xm.is_master_ordinal():
        print_hparams(best_hparams)
        save_hparams(args, best_hparams)
    set_seed(best_hparams["seed"])
    loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
    model_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weights, model_params)
    optimizer = create_optimizer(model, best_hparams, args.loss_type)
    finetune(args, model, loss_fn, optimizer, id_dataset, input_key, spawn_required=False)
    return


def auto_ft(args, model, id_dataset, ood_hp_dataset, max_evals, input_key):
    """Automated fine-tuning using Optuna."""
    PASSWORD = "robust-ft"
    IP_ADDRESS = "10.66.112.3"
    USER = "root"
    DATABASE_NAME = "optuna"
    storage_url = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{IP_ADDRESS}/{DATABASE_NAME}"
    get_or_create_study(study_name=args.save, storage_url=storage_url, direction="minimize", resume_study=args.resume_study)
    xmp.spawn(_mp_auto_ft, args=(args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key,), nprocs=8, start_method='spawn')
    # _mp_auto_ft(0, args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key)
