import copy
import json
import logging
import os
import time

import numpy as np
import optuna
import torch
from src.losses.layerloss import LayerLoss
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.train_utils import evaluate_hp, print_hparams, save_hparams
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('main')
devices = list(range(torch.cuda.device_count()))

def build_hparams_space(trial, num_losses):
    """Build the hyperparameter search space for Optuna."""
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    wd = trial.suggest_float("wd", 0.0, 1.0)
    loss_weights = [trial.suggest_loguniform(f"lossw_{i}", 1e-5, 1e-2) for i in range(num_losses)]
    hparams = {
        "lr": lr,
        "wd": wd,
    }
    for i, loss_weight in enumerate(loss_weights):
        hparams[f"lossw_{i}"] = loss_weight
    return hparams

def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    for _ in range(args.repeats):
        start_time = time.time()
        if args.distributed:
            rank = torch.distributed.get_rank()
            initial_net = DDP(copy.deepcopy(net).to(rank), device_ids=[rank])
            current_net = DDP(copy.deepcopy(net).to(rank), device_ids=[rank])
        else:
            initial_net = copy.deepcopy(net).cuda()
            current_net = copy.deepcopy(net).cuda()
        # initial_state_dict = net.state_dict()
        # current_net = net.cuda().cuda()
        print(f"Time to copy net: {time.time() - start_time:.3f}", flush=True)

        # optimizer = torch.optim.SGD(current_net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"])
        optimizer = torch.optim.AdamW(current_net.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
        loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(args.num_losses)])
        start_time = time.time()
        loss_fn = LayerLoss(loss_weight_hparams, initial_net)
        print(f"Time to create loss fn: {time.time() - start_time:.3f}", flush=True)
        start_time = time.time()
        current_net = finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        print(f"Time to finetune: {time.time() - start_time:.3f}", flush=True)

        start_time = time.time()
        val_results = dict()
        loss, accuracy = evaluate_hp(current_net, ood_hp_dataloader)
        print(f"Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        val_results[f"ood_subset_for_hp_loss"] = loss
        val_results[f"ood_subset_for_hp_accuracy"] = accuracy
        all_val_results.append(val_results)

        # current_net.load_state_dict(initial_state)

    return all_val_results

def auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals_range, input_key):
    """Automated fine-tuning process using Optuna."""

    def hp_objective_fn(trial):
        hparams = build_hparams_space(trial, args.num_losses)
        if args.distributed:
            rank = torch.distributed.get_rank()
            _net = DDP(copy.deepcopy(model).to(rank), device_ids=[rank])
        else:
            _net = copy.deepcopy(model)

        # _model_state = model.state_dict()
        # _net = model
        val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, input_key)
        # _model.load_state_dict(_net)
        return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy

    best_hparams = None
    for max_evals in max_evals_range:
        study = optuna.create_study(direction="minimize")
        study.optimize(hp_objective_fn, n_trials=max_evals)
        best_hparams = study.best_params

        if args.distributed:
            rank = torch.distributed.get_rank()
            _model = DDP(copy.deepcopy(model).to(rank), device_ids=[rank])
        else:
            _model = copy.deepcopy(model).cuda()
        # _model_state = model.state_dict()
        # _model = model
        loss_weight_hparams = torch.tensor([best_hparams[f"lossw_{i}"] for i in range(args.num_losses)])
        loss_fn = LayerLoss(loss_weight_hparams, _model)
        optimizer = torch.optim.AdamW(_model.parameters(), lr=best_hparams["lr"], weight_decay=best_hparams["wd"])
        _model = finetune(args, _model, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        start_time = time.time()
        results = evaluate(_model, args)
        print(results, flush=True)
        logger.info(json.dumps(results, indent=4))
        print(f"Time to evaluate: {time.time() - start_time:.3f}", flush=True)
        print_hparams(best_hparams)

    # Saving model
    if args.save is not None and args.rank == 0:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.inner_steps}.pt')
        print('Saving model to', str(model_path), flush=True)
        logger.info(f"Saving model to {str(model_path)}")
        model.save(model_path)
        save_hparams(args, best_hparams)
        return model_path

    return results, best_hparams


# def build_hparams_space(num_losses):
#     """Build the hyperparameter search space."""
#     space = {
#         "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
#         "wd": hp.uniform("wd", 0.0, 1.0),
#         # "momentum": hp.uniform("momentum", 0.0, 1.0)
#     }
#     for i in range(num_losses):
#         space[f"lossw_{i}"] = hp.loguniform(f"lossw_{i}", np.log(1e-5), np.log(1e-2))
#     return space
#
#
# def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
#     """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
#     all_val_results = []
#     for _ in range(args.repeats):
#         start_time = time.time()
#         initial_net = copy.deepcopy(net).cuda()
#         current_net = copy.deepcopy(net).cuda()
#         print(f"Time to copy net: {time.time() - start_time:.3f}")
#         # optimizer = torch.optim.SGD(current_net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"])
#         optimizer = torch.optim.AdamW(current_net.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
#         loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(args.num_losses)])
#         start_time = time.time()
#         loss_fn = LayerLoss(loss_weight_hparams, initial_net)
#         print(f"Time to create loss fn: {time.time() - start_time:.3f}")
#         start_time = time.time()
#         current_net = finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
#         print(f"Time to finetune: {time.time() - start_time:.3f}")
#
#         start_time = time.time()
#         val_results = dict()
#         loss, accuracy = evaluate_hp(current_net, ood_hp_dataloader)
#         print(f"Time to evaluate: {time.time() - start_time:.3f}")
#         val_results[f"ood_subset_for_hp_loss"] = loss
#         val_results[f"ood_subset_for_hp_accuracy"] = accuracy
#         all_val_results.append(val_results)
#
#     return all_val_results
#
# def auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals_range, input_key):
#     """Automated fine-tuning process."""
#     def hp_objective_fn(hparams):
#         _net = copy.deepcopy(model)
#         val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, input_key)
#         return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy
#
#     space = build_hparams_space(args.num_losses)
#     trials = Trials()
#
#     best_hparams = None
#     for max_evals in max_evals_range:
#         best_hparams = fmin(fn=hp_objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
#         _model = copy.deepcopy(model)
#         loss_weight_hparams = torch.tensor([best_hparams[f"lossw_{i}"] for i in range(args.num_losses)])
#         loss_fn = LayerLoss(loss_weight_hparams, _model)
#         # optimizer = torch.optim.SGD(_model.parameters(), lr=best_hparams["lr"], momentum=best_hparams["momentum"])
#         optimizer = torch.optim.AdamW(_model.parameters(), lr=best_hparams["lr"], weight_decay=best_hparams["wd"])
#         _model = finetune(args, _model, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
#         results = evaluate(_model, args)
#         print_hparams(best_hparams)
#
#     # Saving model
#     if args.save is not None:
#         os.makedirs(args.save, exist_ok=True)
#         model_path = os.path.join(args.save, f'checkpoint_{args.inner_steps}.pt')
#         print('Saving model to', str(model_path))
#         model.save(model_path)
#         save_hparams(args, best_hparams)
#         return model_path
#
#     return results, best_hparams