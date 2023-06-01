#%%
import argparse
import copy
import random
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from learned_optimizer import fine_tune, metalearn_opt, sample_ft_data
from models import get_pretrained_net, pretrain_nets
from sine import savefig


def sanity_check(ckpt_path, task_distribution):
    """Sanity check with full vs surgical fine-tuning on only last layer."""
    meta_params = torch.ones(4).float() * 100
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.tensor([-100, -100, -100, 100]).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")


def run_expt(args):
    pretrain_nets(ckpt_path=args.ckpt_path, num_nets=args.num_nets)
    sanity_check(ckpt_path=args.ckpt_path, task_distribution=args.task_distribution)
    start = time.time()
    meta_params = metalearn_opt(args)
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")

    # Plot single task
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        task_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
        train_x, train_y, test_x, test_y = task_data
        full_x = torch.tensor(np.linspace(0, 2 * np.pi, 100)).reshape(-1, 1).float()

        full_ft_net, _ = fine_tune(args.optimizer_name, net, torch.zeros(4), task_data)
        meta_ft_net, _ = fine_tune(args.optimizer_name, net, meta_params, task_data)
        initial_preds = net(full_x).detach()
        full_ft_preds = full_ft_net(full_x).detach()
        meta_ft_preds = meta_ft_net(full_x).detach()

        ax.scatter(train_x, train_y, c="k", label="train", zorder=10, s=50)
        ax.plot(full_x, initial_preds, c="gray", label="pretrained")
        ax.plot(full_x, full_ft_preds, c="red", label="full ft")
        ax.plot(full_x, meta_ft_preds, c="blue", label="meta+ft")
        ax.scatter(test_x, test_y, c="gray", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    savefig(f"meta_ft_{args.task_distribution}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_distribution",
        type=str,
        default="bias_shift",
        choices=["amp_shift", "bias_shift", "amp_bias_shift"],
    )
    parser.add_argument("--optimizer_name", type=str, default="LayerSGDLinear", choices=["LayerSGD", "LayerSGDLinear"])
    parser.add_argument("--meta_steps", type=int, default=50)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--val_meta_batch_size", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--train_N", type=int, default=10)
    parser.add_argument("--ckpt_path", type=str, default="/iris/u/yoonho/robust-optimizer/ckpts/sine")
    parser.add_argument("--num_nets", type=int, default=100)
    args = parser.parse_args()

    run_expt(args)
# %%
