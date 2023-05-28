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

from sine import generate_sine_data, savefig


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


ckpt_path = Path("/iris/u/yoonho/robust-optimizer/ckpts/sine")
ckpt_path.mkdir(exist_ok=True)
for seed in range(100):
    filename = ckpt_path / f"pretrain_{seed}.pt"
    if not filename.exists():
        net = pretrain_net(seed=seed)
        torch.save(net.state_dict(), filename)
        print(f"Saved pretrained net to {filename}!")

class LayerSGD(Optimizer):
    """ meta-params: pre-sigmoid lr_multiplier per parameter. """

    def __init__(self, meta_params, params, lr=required):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.meta_params = meta_params

    @staticmethod
    def get_init_meta_params():
        return torch.zeros(4)

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

class LayerSGDLinear(Optimizer):
    """ meta-params: weights of linear layer with depth as input. """
    def __init__(self, meta_params, net, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list([p for p in net.children() if isinstance(p, nn.Linear)])  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w"})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b"})
        super().__init__(param_groups, defaults)
        self.meta_params = {"w": meta_params[0:2], "b": meta_params[2:4]}

    @staticmethod
    def get_init_meta_params():
        return torch.zeros(4)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        return super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue

            depth = group["depth"]
            meta_params = self.meta_params[group["type"]]
            lr_multiplier = torch.sigmoid(meta_params[0] * depth + meta_params[1])
            p.data.add_(p.grad.data, alpha=-group["lr"] * lr_multiplier)
        return loss


def get_pretrained_net(train):
    """Return a randomly sampled pretrained net."""
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


def fine_tune(optimizer_name, net, meta_params, ft_data, inner_steps=10, inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net and test loss."""
    net = copy.deepcopy(net)
    train_x, train_y, test_x, test_y = ft_data
    if optimizer_name == "LayerSGD":
        inner_opt = LayerSGD(meta_params, net.parameters(), lr=inner_lr)
    elif optimizer_name == "LayerSGDLinear":
        inner_opt = LayerSGDLinear(meta_params, net, lr=inner_lr)
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


def sample_ft_data(task_distribution, train_N=10):
    """Sample fine-tuning data."""
    if task_distribution == "amp_shift":
        amplitude = np.random.uniform(-2, 2)
        bias = 0.0
    elif task_distribution == "bias_shift":
        amplitude = 1.0
        bias = np.random.uniform(-2, 2)
    elif task_distribution == "amp_bias_shift":
        amplitude = np.random.uniform(-2, 2)
        bias = np.random.uniform(-2, 2)
    else:
        raise ValueError(f"Unknown task distribution {task_distribution}")
    ft_data = generate_sine_data(
        train_range=(0.0, 2.0),
        test_range=(0, 2 * np.pi),
        N=train_N,
        amplitude=amplitude,
        bias=bias,
    )
    train_x, train_y = ft_data["train"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)
    test_x, test_y = ft_data["test"]
    test_x = torch.tensor(test_x).float().reshape(-1, 1)
    test_y = torch.tensor(test_y).float().reshape(-1, 1)
    return train_x, train_y, test_x, test_y


def sanity_check(task_distribution):
    """Sanity check with full vs surgical fine-tuning on only last layer."""
    meta_params = torch.ones(4).float() * 100
    losses = []
    for _ in range(100):
        net = get_pretrained_net(train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.tensor([-100, -100, -100, 100]).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune("LayerSGD", net, meta_params, ft_data, inner_steps=10, inner_lr=0.1)
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")


def metalearn_opt(args):
    if args.optimizer_name == "LayerSGD":
        meta_params = LayerSGD.get_init_meta_params()
    elif args.optimizer_name == "LayerSGDLinear":
        meta_params = LayerSGDLinear.get_init_meta_params()
    meta_optimizer = torch.optim.SGD([meta_params], lr=args.meta_lr)

    for meta_step in range(args.meta_steps + 1):
        grads = []
        metrics = defaultdict(list)
        if meta_step % args.val_freq == 0:
            for _ in range(args.val_meta_batch_size):
                net = get_pretrained_net(train=False)
                ft_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
                _, current_loss = fine_tune(args.optimizer_name, net, meta_params, ft_data)
                metrics["loss"].append(current_loss)
            current_loss = np.array(metrics["loss"])
            print(
                f"Meta-step {meta_step}: current loss: {current_loss.mean():.3f}+-{current_loss.std():.3f}"
            )

        for _ in range(args.meta_batch_size):
            net = get_pretrained_net(train=True)
            ft_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
            epsilon = torch.randn(size=meta_params.shape)  # Antithetic sampling
            mp_plus_epsilon = meta_params + epsilon * args.noise_std
            mp_minus_epsilon = meta_params - epsilon * args.noise_std
            _, loss_plus = fine_tune(args.optimizer_name, net, mp_plus_epsilon, ft_data)
            _, loss_minus = fine_tune(args.optimizer_name, net, mp_minus_epsilon, ft_data)
            grads.append((loss_plus - loss_minus) * epsilon / args.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        meta_optimizer.zero_grad()
        meta_params.grad = grads_mean
        meta_optimizer.step()
    print(f"Final meta-params: {meta_params}")
    return meta_params


def run_expt(args):
    # sanity_check(args)
    start = time.time()
    meta_params = metalearn_opt(args)
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")

    # Plot single task
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        net = get_pretrained_net(train=False)
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
    parser.add_argument("--meta_batch_size", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--val_meta_batch_size", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--train_N", type=int, default=10)
    args = parser.parse_args()

    run_expt(args)
# %%
