#%%
import argparse
import copy
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from learned_optimizer import fine_tune, metalearn_opt
from models import get_pretrained_net, pretrain_nets
from mnist import load_mnist


def sanity_check(ckpt_path, task_distribution):
    """Sanity check with full vs surgical fine-tuning on only last layer."""
    meta_params = torch.ones(6).float() * 100
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        train_loader, test_loader = load_mnist(task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, train_loader, test_loader, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.ones(6).float() * 100
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        train_loader, test_loader = load_mnist(dataset=task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, train_loader, test_loader, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")


def run_expt(args):
    pretrain_nets(ckpt_path=args.ckpt_path, num_nets=args.num_nets)
    # sanity_check(ckpt_path=args.ckpt_path, task_distribution=args.task_distribution)
    start = time.time()
    meta_params = metalearn_opt(args)
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_distribution",
        type=str,
        default="mnistc",
        choices=["mnist", "mnistc", "emnist", "kmnist", "mnist-label-shift"],
    )
    parser.add_argument("--optimizer_name", type=str, default="LayerSGDLinear", choices=["LayerSGD", "LayerSGDLinear"])
    parser.add_argument("--meta_steps", type=int, default=5)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--val_meta_batch_size", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--train_N", type=int, default=10)
    parser.add_argument("--ckpt_path", type=str, default="/iris/u/cchoi1/robust-optimizer/ckpts/mnist")
    parser.add_argument("--num_nets", type=int, default=10)
    args = parser.parse_args()

    print("EXPERIMENT ARGS")
    print(args)

    run_expt(args)
# %%
