from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets

from src.losses.layerloss import LayerLoss
from src.models.train_utils import evaluate_hp, print_hparams, save_hparams, set_seed, get_subset
import numpy as np
from torch.utils.data import DataLoader
from hyperopt import fmin, hp, tpe, Trials
from src.models.finetune import finetune
from src.models.eval import evaluate


def build_hparams_space(num_losses):
    """Build the hyperparameter search space."""
    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        "wd": hp.uniform("wd", 0.0, 1.0),
        # "momentum": hp.uniform("momentum", 0.0, 1.0)
    }
    for i in range(num_losses):
        space[f"lossw_{i}"] = hp.loguniform(f"lossw_{i}", np.log(1e-5), np.log(1e-2))
    return space


def evaluate_hparams(args, net, hyperparams, id_dataloader, ood_hp_dataloader, input_key):
    """Evaluate a given set of hyperparameters on ood_subset_for_hp."""
    all_val_results = []
    for _ in range(args.repeats):
        start_time = time.time()
        initial_net = copy.deepcopy(net).cuda()
        current_net = copy.deepcopy(net).cuda()
        print(f"Time to copy net: {time.time() - start_time:.3f}")
        # optimizer = torch.optim.SGD(current_net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"])
        optimizer = torch.optim.AdamW(current_net.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
        loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(args.num_losses)])
        start_time = time.time()
        loss_fn = LayerLoss(loss_weight_hparams, initial_net)
        print(f"Time to create loss fn: {time.time() - start_time:.3f}")
        start_time = time.time()
        current_net = finetune(args, current_net, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        print(f"Time to finetune: {time.time() - start_time:.3f}")

        start_time = time.time()
        val_results = dict()
        loss, accuracy = evaluate_hp(current_net, ood_hp_dataloader)
        print(f"Time to evaluate: {time.time() - start_time:.3f}")
        val_results[f"ood_subset_for_hp_loss"] = loss
        val_results[f"ood_subset_for_hp_accuracy"] = accuracy
        all_val_results.append(val_results)

    return all_val_results


# def report_evaluation(all_val_results, eval_datasets):
#     """Report the evaluation results."""
#     test_accs = []
#     for name in eval_datasets:
#         losses = [r[f"{name}_loss"] for r in all_val_results]
#         accs = [r[f"{name}_accuracy"] for r in all_val_results]
#         if "test" in name:
#             test_accs.append(np.mean(accs))
#         print(f"{name:10s} loss: {np.mean(losses):.3f} ± {np.std(losses):.3f}  acc: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
#     print(f"Average Test Accuracy: {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}")
#     print()

def auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals_range, input_key):
    """Automated fine-tuning process."""
    def hp_objective_fn(hparams):
        _net = copy.deepcopy(model)
        val_results = evaluate_hparams(args, _net, hparams, id_dataloader, ood_hp_dataloader, input_key)
        return -np.mean([r["ood_subset_for_hp_accuracy"] for r in val_results])  # maximize accuracy

    space = build_hparams_space(args.num_losses)
    trials = Trials()

    best_hparams = None
    for max_evals in max_evals_range:
        best_hparams = fmin(fn=hp_objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        _model = copy.deepcopy(model)
        loss_weight_hparams = torch.tensor([best_hparams[f"lossw_{i}"] for i in range(args.num_losses)])
        loss_fn = LayerLoss(loss_weight_hparams, _model)
        optimizer = torch.optim.SGD(_model.parameters(), lr=best_hparams["lr"], momentum=best_hparams["momentum"])
        _model = finetune(args, _model, loss_fn, optimizer, id_dataloader, input_key, print_every=None)
        results = evaluate(_model, args)
        print_hparams(best_hparams)

    # Saving model
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.inner_steps}.pt')
        print('Saving model to', str(model_path))
        model.save(model_path)
        save_hparams(args, best_hparams)
        return model_path

    return results, best_hparams