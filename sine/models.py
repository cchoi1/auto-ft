import argparse
import copy
import random
import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sine import generate_sine_data


def get_network(hidden=50):
    net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1))
    return net


def pretrain_net(seed=0, lr=1e-3, num_steps=3000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = get_network()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    data = generate_sine_data(N=1000)
    train_x, train_y = data["train"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)

    loss_fn = nn.MSELoss()
    for _ in range(num_steps):
        preds = net(train_x)
        loss = loss_fn(preds, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Final loss: {loss.item():.4f}")
    return net

def pretrain_nets(ckpt_path, num_nets):
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(exist_ok=True)
    for seed in range(num_nets):
        filename = ckpt_path / f"pretrain_{seed}.pt"
        if not filename.exists():
            net = pretrain_net(seed=seed)
            torch.save(net.state_dict(), filename)
            print(f"Saved pretrained net to {filename}!")

def get_pretrained_net(ckpt_path, train):
    """Return a randomly sampled pretrained net."""
    ckpt_path = Path(ckpt_path)
    all_ckpts = glob(str(ckpt_path / "pretrain_*.pt"))
    n_ckpts = len(all_ckpts)
    train_N = int(n_ckpts * 0.8)
    train_ckpts, test_ckpts = all_ckpts[:train_N], all_ckpts[train_N:]
    if train:
        random_fn = random.choice(train_ckpts)
    else:
        random_fn = random.choice(test_ckpts)
    rand_checkpoint = torch.load(random_fn)
    net = get_network()
    net.load_state_dict(rand_checkpoint)
    return net