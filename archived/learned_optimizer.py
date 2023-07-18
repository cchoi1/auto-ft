"""Old code for meta-training several randomly-initialized models in parallel."""

import copy
import importlib
from collections import defaultdict
from functools import partial

import functorch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets import get_dataloaders
from networks import get_pretrained_net, get_pretrained_net_fixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_fn(num_nets: int, ckpt_path: str, train: bool):
    """Combines the states of several pretrained nets together by stacking each parameter.
    Returns a stateless version of the model (func_net) and stacked parameters and buffers."""
    nets = []
    for _ in range(num_nets):
        net = copy.deepcopy(get_pretrained_net(ckpt_path=ckpt_path, train=train))
        nets.append(net)
    func_net, batched_weights, buffers = functorch.combine_state_for_ensemble(nets)
    return func_net, batched_weights, buffers

def fine_tune_func_n(optimizer_obj, inner_steps, inner_lr, func_net, buffers, net_params, meta_params, train_images, train_labels, test_images, test_labels):
    """Fine-tune func_net on (train_images, train_labels), and return test losses.
    In the outer loop, we use vmap to parallelize calls to this function for each task in the meta-batch.
    Params:
        func_net: batched functional net (i.e. (args.meta_batch_size // 2) randomly sampled pretrained models)
        buffers: buffers needed to call forward() on the batched, functional model func_net
        net_params: batched parameters (i.e. (args.meta_batch_size // 2) randomly sampled pretrained models)"""
    inner_opt = optimizer_obj(meta_params=meta_params, params=net_params, lr=inner_lr)
    def compute_stateless_loss(params, inputs, labels):
        outputs = func_net(params, buffers, inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    test_losses = []
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        gradients = torch.func.grad(compute_stateless_loss)(net_params, train_images, train_labels)
        net_params = inner_opt.update(net_params, gradients)

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = func_net(net_params, buffers, test_images)
        test_loss = F.cross_entropy(outputs, test_labels) # (meta_batch_size // 2, 1)
        test_losses.append(test_loss)

    return test_losses