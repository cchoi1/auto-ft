#%%
import argparse
import random
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.optimizer import Optimizer, required

from sine import generate_sine_data


def get_network(hidden=50):
    net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1))
    return net


def pretrain_net(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = get_network()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    data = generate_sine_data(N=1000)
    train_x, train_y = data["train"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)

    loss_fn = nn.MSELoss()
    for i in range(3000):
        preds = net(train_x)
        loss = loss_fn(preds, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Final loss: {loss.item():.4f}")
    return net


ckpt_path = Path("/iris/u/yoonho/robust-optimizer/ckpts/sine")
ckpt_path.mkdir(exist_ok=True)
for seed in range(100):
    filename = ckpt_path / f"pretrain_{seed}.pt"
    if not filename.exists():
        net = pretrain_net(seed=seed)
        torch.save(net.state_dict(), filename)
        print(f"Saved pretrained net to {filename}!")

# %% Start meta-learning
class LayerSGD(Optimizer):
    """meta-params: lr_multiplier per layer"""

    def __init__(self, meta_params, params, lr=required):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.meta_params = meta_params

    def __setstate__(self, state: Dict[str, Any]) -> None:
        return super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                lr_multiplier = torch.sigmoid(self.meta_params[i])
                local_lr = group["lr"] * lr_multiplier
                p.data.add_(d_p, alpha=-local_lr)
        return loss


def get_pretrained_net():
    all_checkpoint_paths = glob(str(ckpt_path / "pretrain_*.pt"))
    random_fn = random.choice(all_checkpoint_paths)
    rand_checkpoint = torch.load(random_fn)
    net = get_network()
    net.load_state_dict(rand_checkpoint)
    return net

def fine_tune(net, meta_params, ft_data, inner_steps=10, inner_lr=1e-1):
    import copy

    net = copy.deepcopy(net)
    train_x, train_y, test_x, test_y = ft_data
    inner_opt = LayerSGD(meta_params, net.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()
    test_losses = []
    for _ in range(inner_steps):
        preds = net(train_x)
        loss = loss_fn(preds, train_y)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

        test_preds = net(test_x)
        test_loss = loss_fn(test_preds, test_y)
        test_losses.append(test_loss.item())
    return net, np.mean(test_losses)


def sample_ft_data(change_amplitude: bool, change_vertical_shift: bool):
    amplitude = 1.0
    vertical_shift = 0.0
    if change_amplitude:
        amplitude = np.random.uniform(-2, 2)
    if change_vertical_shift:
        vertical_shift = np.random.uniform(-2, 2)
    ft_data = generate_sine_data(
        train_range=(0.0, 2.0), test_range=(0, 2 * np.pi), N=10, amplitude=amplitude, vertical_shift=vertical_shift
    )
    train_x, train_y = ft_data["train"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)
    test_x, test_y = ft_data["test"]
    test_x = torch.tensor(test_x).float().reshape(-1, 1)
    test_y = torch.tensor(test_y).float().reshape(-1, 1)
    return train_x, train_y, test_x, test_y

def sanity_check(change_amplitude: bool, change_vertical_shift: bool):
    """
    Sanity check with full vs surgical fine-tuning on only last layer.
    """
    meta_params = torch.ones(4).float() * -100
    losses = []
    for _ in range(100):
        net = get_pretrained_net()
        ft_data = sample_ft_data(change_amplitude, change_vertical_shift)
        _, loss = fine_tune(net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.tensor([-100, -100, -100, 100]).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net()
        ft_data = sample_ft_data(change_amplitude, change_vertical_shift)
        _, loss = fine_tune(net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")

def metalearn_opt(change_amplitude: bool, change_vertical_shift: bool):
    # Evolution strategies hyperparameters
    meta_steps = 100
    meta_batch_size = 20
    noise_std = 1.0
    meta_lr = 1e-2

    meta_params = torch.zeros(4)
    meta_optimizer = torch.optim.SGD([meta_params], lr=meta_lr)

    for _ in range(1, meta_steps + 1):
        grads = []
        metrics = defaultdict(list)
        for _ in range(meta_batch_size):
            net = get_pretrained_net()
            ft_data = sample_ft_data(change_amplitude, change_vertical_shift)
            _, current_loss = fine_tune(net, meta_params, ft_data)
            metrics["loss"].append(current_loss)

            epsilon = torch.randn(size=meta_params.shape)
            mp_plus_epsilon = meta_params + epsilon * noise_std
            mp_minus_epsilon = meta_params - epsilon * noise_std
            _, loss_plus = fine_tune(net, mp_plus_epsilon, ft_data)
            _, loss_minus = fine_tune(net, mp_minus_epsilon, ft_data)
            grads.append((loss_plus - loss_minus) * epsilon / noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        current_loss = np.array(metrics["loss"])
        print(f"Current loss: {current_loss.mean():.2f}+-{current_loss.std():.2f}")

        meta_optimizer.zero_grad()
        meta_params.grad = grads_mean
        meta_optimizer.step()

    print(f"Final meta-params: {meta_params}")
    return meta_params

def run_expt(args):
    sanity_check(change_amplitude=args.change_amplitude, change_vertical_shift=args.change_vertical_shift)

    meta_params = metalearn_opt(change_amplitude=args.change_amplitude, change_vertical_shift=args.change_vertical_shift)

    shift_type = ""
    if args.change_amplitude:
        shift_type += "amplitude_"
    if args.change_vertical_shift:
        shift_type += "bias_"

    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        net = get_pretrained_net()
        task_data = sample_ft_data(change_amplitude=args.change_amplitude, change_vertical_shift=args.change_vertical_shift)
        train_x, train_y, test_x, test_y = task_data
        full_x = torch.tensor(np.linspace(0, 2 * np.pi, 100)).reshape(-1, 1).float()

        full_ft_net, _ = fine_tune(net, torch.zeros(4), task_data)
        meta_ft_net, _ = fine_tune(net, meta_params, task_data)
        initial_preds = net(full_x).detach()
        full_ft_preds = full_ft_net(full_x).detach()
        meta_ft_preds = meta_ft_net(full_x).detach()

        ax.scatter(train_x, train_y, c="k", label="train", zorder=10, s=50)
        ax.plot(full_x, initial_preds, c="gray", label="pretrained")
        ax.plot(full_x, full_ft_preds, c="red", label="full ft")
        ax.plot(full_x, meta_ft_preds, c="blue", label="meta+ft")
        ax.scatter(test_x, test_y, c="gray", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    from sine import savefig
    savefig(f"meta_ft_{shift_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--change_amplitude", action='store_true')
    parser.add_argument("--change_vertical_shift", action='store_true')
    args = parser.parse_args()
    run_expt(args)