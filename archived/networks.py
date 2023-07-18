"""Old code for sampling a random pretrained model."""

import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datasets import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pretrained_net(ckpt_path, dataset_name, train):
    """Return a randomly sampled pretrained net."""
    ckpt_path = Path(ckpt_path)
    all_ckpts = glob(str(ckpt_path / f"pretrain_{dataset_name}*.pt"))
    n_ckpts = len(all_ckpts)
    train_N = int(n_ckpts * 0.8)
    train_ckpts, test_ckpts = all_ckpts[:train_N], all_ckpts[train_N:]
    if train:
        random_fn = np.random.choice(train_ckpts)
    else:
        random_fn = np.random.choice(test_ckpts)
    rand_checkpoint = torch.load(random_fn)
    net = get_network()
    net.load_state_dict(rand_checkpoint)
    net = net.to(device)
    return net