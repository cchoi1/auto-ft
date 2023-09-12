import copy
import gc
import json
import logging
import os

import numpy as np
import optuna
import src.losses as losses
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader
from src.models.finetune import finetune_helper, finetune
from src.models.modeling import ImageClassifier
from src.models.utils import set_seed, initialize_model

logger = logging.getLogger('main')

@torch.no_grad()
def evaluate_net(net, dataloader):
    """Evaluate the given model on a dataloader."""
    losses = []
    total_samples, correct = 0, 0
    i = 0
    for batch in dataloader:
        i += 1
        if isinstance(batch, dict):
            x = batch["images"]
            y = batch["labels"]
        else:
            x, y = batch

        output = net(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum()
        total_samples += x.size()[0]
        loss = F.cross_entropy(output, y)
        losses.append(loss)

        del batch; del x; del y; del output; del pred; del loss
        gc.collect()

    xm.master_print('finished epoch in evaluate_net')
    loss = xm.mesh_reduce("loss_mean", torch.stack(losses).mean().item(), np.mean)
    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce("ood_hp_accuracy", accuracy, np.mean)
    del losses; del total_samples; del correct
    gc.collect()

    return loss, accuracy


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


def get_or_create_study(study_name, storage_url, direction="minimize", load_existing_study=False):
    """
    Load a study if it exists, otherwise create a new one.

    Parameters:
    - study_name (str): Name of the study.
    - storage_url (str): Storage URL for the database.
    - direction (str): Optimization direction ("minimize" or "maximize").

    Returns:
    - study: An Optuna study object.
    """
    all_study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    existing_study_names = [summary.study_name for summary in all_study_summaries]
    if study_name in existing_study_names:
        if not load_existing_study:
            msg = f"Found existing study {study_name}. Deleting it and creating a new one."
            xm.master_print(msg)
            logger.info(msg)
            optuna.delete_study(study_name=study_name, storage=storage_url)
            study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction)
        else:
            msg = f"Found existing study {study_name}. Loading it."
            xm.master_print(msg)
            logger.info(msg)
            study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        msg = f"Could not find existing study {study_name}. Creating a new one."
        xm.master_print(msg)
        logger.info(msg)
        study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction)
    return


def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    device = xm.xla_device()
    for _ in range(args.repeats):
        current_net = copy.deepcopy(net)
        set_seed(hyperparams["seed"])
        optimizer = create_optimizer(current_net, hyperparams, args.loss_type)
        initial_net_params = [p for p in net.parameters()]
        loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)

        current_net = finetune_helper(args, device, current_net, loss_fn, optimizer, id_dataloader, input_key, steps=args.inner_steps)
        del id_dataloader; del optimizer; del loss_fn; del initial_net_params; del loss_weight_hparams
        gc.collect()

        val_results = dict()
        loss, accuracy = evaluate_net(current_net, ood_hp_dataloader)
        val_results[f"ood_subset_for_hp_loss"] = loss
        val_results[f"ood_subset_for_hp_accuracy"] = accuracy
        all_val_results.append(val_results)
        del ood_hp_dataloader; del current_net
        gc.collect()

    return all_val_results


def _mp_auto_ft(rank, args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key):
    """Automated fine-tuning using Optuna."""
    xm.master_print("Starting _mp_auto_ft")
    torch.manual_seed(args.seed + rank)
    device = xm.xla_device()
    model = model.to(device)
    id_dataloader = get_dataloader(id_dataset, is_train=True, args=args, image_encoder=None)
    ood_hp_dataloader = get_dataloader(ood_hp_dataset, is_train=True, args=args, image_encoder=None)
    def hp_objective_fn(trial):
        hparams = HyperparameterSpace(model, args.loss_type, args.num_losses).build_space(trial)
        val_results = evaluate_hparams(args, model, hparams, id_dataloader, ood_hp_dataloader, input_key)
        objective = -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])
        del val_results
        gc.collect()
        # xm.mark_step()
        return objective

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        xm.master_print("Starting hyperparameter optimization")
        study = optuna.load_study(study_name=args.save, storage=storage_url)
        xm.master_print("Loaded study")
        study.optimize(hp_objective_fn, n_trials=max_evals, callbacks=[clear_memory])
        xm.master_print("Finished hyperparameter optimization")
        best_hparams = study.best_params
        print_hparams(best_hparams)
        save_hparams(args, best_hparams)

    ft_model = copy.deepcopy(model)
    set_seed(best_hparams["seed"])
    loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
    model_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weights, model_params)
    optimizer = create_optimizer(ft_model, best_hparams, args.loss_type)
    finetune(args, model, loss_fn, optimizer, id_dataset, input_key, spawn_required=False)

def auto_ft(args, model, id_dataset, ood_hp_dataset, max_evals, input_key):
    """Automated fine-tuning using Optuna."""
    PASSWORD = "robust-ft"
    IP_ADDRESS = "10.66.112.3"
    USER = "root"
    DATABASE_NAME = "optuna"
    storage_url = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{IP_ADDRESS}/{DATABASE_NAME}"
    get_or_create_study(study_name=args.save, storage_url=storage_url, direction="minimize", load_existing_study=args.load_existing_study)
    _mp_auto_ft(0, args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key)
    # xmp.spawn(_mp_auto_ft, args=(args, model, id_dataset, ood_hp_dataset, storage_url, max_evals, input_key,), nprocs=8, start_method='spawn')
