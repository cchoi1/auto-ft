import copy
import gc
import json
import logging
import os
import time

import numpy as np
from hyperopt import hp, fmin, tpe, Trials
import torch

import src.losses as losses
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.finetune import inner_finetune, finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel, set_seed

logger = logging.getLogger('main')

@torch.no_grad()
def evaluate_net(net, dataloader):
    total_correct = 0
    total_samples = 0
    net = net.cuda()
    net.eval()
    for batch in dataloader:
        data = maybe_dictionarize(batch)
        x = data['images'].cuda()
        y = data['labels'].cuda()
        outputs = net(x)

        if (y == -1).any():  # Handle unlabeled parts of the batch
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (y == -1)
            y[mask] = pseudo_labels[mask]

        predictions = outputs.argmax(dim=1)
        correct = (predictions == y).sum().item()
        total_correct += correct
        total_samples += y.size(0)

    accuracy = (total_correct / total_samples) * 100
    torch.cuda.empty_cache()
    return accuracy


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
    def __init__(self, model, loss_type, num_losses=None, num_datapoints=None):
        self.model = model
        self.loss_type = loss_type
        self.num_losses = num_losses
        self.num_datapoints = num_datapoints
        if loss_type == "LayerwiseLoss":
            assert num_losses is not None

    def _base_space(self, prefix):
        return {
            f"{prefix}lr": hp.loguniform(f"{prefix}lr", np.log(1e-7), np.log(1e-3)),
            f"{prefix}wd": hp.uniform(f"{prefix}wd", 0.0, 1.0),
            **{f"{prefix}lossw_{i}": hp.loguniform(f"{prefix}lossw_{i}", np.log(1e-5), np.log(1e-2)) for i in range(self.num_losses)}
        }

    def build_learned_loss_space(self):
        hparams = self._base_space("")
        hparams["seed"] = hp.choice("seed", list(range(0, 101))) # Using hp.choice to simulate the integer range for seed
        if self.num_datapoints:
            hparams.update({f"dataw_{i}": hp.uniform(f"dataw_{i}", 0.0, 1.0) for i in range(self.num_datapoints)})
        return hparams

    def build_layerwise_loss_space(self):
        hparams = {}
        layer_idx = 0
        self.model = extract_from_data_parallel(self.model)
        for name, module in self.model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    hparams.update(self._base_space(f"{layer_idx}_"))
                    layer_idx += 1
                # Classification head of the model
                hparams.update(self._base_space(f"{layer_idx}_"))
        hparams["seed"] = hp.choice("seed", list(range(0, 101)))  # Using hp.choice to simulate the integer range for seed
        return hparams

    def build_space(self):
        if self.loss_type == "LayerwiseLoss":
            return self.build_layerwise_loss_space()
        else:
            assert self.num_losses is not None
            return self.build_learned_loss_space()

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

def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key, id_val_dataloader=None):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    start_time = time.time()
    all_val_results = []
    for _ in range(args.repeats):
        current_net = copy.deepcopy(net)
        print(f"Time to copy: {time.time() - start_time:.3f}", flush=True)
        start_time = time.time()
        optimizer = create_optimizer(current_net, hyperparams, args.loss_type)
        initial_net_params = [p for p in net.parameters()]
        loss_weight_hparams = get_loss_weights(hyperparams, args.loss_type, args.num_losses)
        loss_fn = getattr(losses, args.loss_type)(loss_weight_hparams, initial_net_params)
        print(f"Time to construct loss: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        if args.pointwise_loss and "dataw_0" in hyperparams.keys():
            data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(id_dataloader.dataset))])
            sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(id_dataloader.dataset))
            id_dataloader = get_dataloader(id_dataloader.dataset, is_train=True, args=args, sampler=sampler, image_encoder=None)
        set_seed(hyperparams["seed"])
        if args.use_id_val:
            current_net = inner_finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None, id_val_acc_thresh=0.96, id_val_dataloader=id_val_dataloader)
        else:
            current_net = inner_finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        set_seed(args.seed)
        print(f"Time to finetune: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        val_results = dict()
        ood_accuracy = evaluate_net(current_net, ood_hp_dataloader)
        print(f"Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        val_results[f"ood_subset_for_hp_accuracy"] = ood_accuracy
        all_val_results.append(val_results)
    print(f"Time to evaluate hparams: {time.time() - start_time:.3f}", flush=True)

    return all_val_results

def auto_ft_hyperopt(args, model, id_dataloader, ood_hp_dataloader, max_evals, input_key, print_every=None, id_val_dataloader=None):
    """Automated fine-tuning process using hyperopt."""
    num_datapoints = len(id_dataloader.dataset) if args.pointwise_loss else None
    space = HyperparameterSpace(model, args.loss_type, args.num_losses, num_datapoints).build_space()
    def hp_objective_fn(hparams):
        _net = copy.deepcopy(model).cuda()
        val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, input_key, id_val_dataloader)
        return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy

    if args.load_hparams is not None:
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
    else:
        trials = Trials()
        best_hparams = fmin(fn=hp_objective_fn,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)
        for trial in trials.trials:
            print("Trial:", trial['tid'])
            print("Hyperparameters:", trial['misc']['vals'])
            print("Result:", trial['result'])
            print("------------")
    loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
    initial_params = [p for p in model.parameters()]
    loss_fn = getattr(losses, args.loss_type)(loss_weights, initial_params)
    optimizer = create_optimizer(model, best_hparams, args.loss_type)
    print_hparams(best_hparams)
    save_hparams(best_hparams, args)
    print_every = 100 if args.plot else None
    id_dataset = id_dataloader.dataset
    del id_dataloader; torch.cuda.empty_cache()
    set_seed(best_hparams["seed"])
    ft_model = finetune_final(args, model, loss_fn, optimizer, id_dataset, input_key, print_every=print_every)

    return ft_model