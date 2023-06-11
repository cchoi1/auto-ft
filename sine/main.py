import argparse
import copy
import importlib
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from networks import get_pretrained_net, pretrain_nets
from tasks import generate_sine_data, savefig
from torch import nn


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


def fine_tune(optimizer_obj, net, meta_params, ft_data, inner_steps=10, inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net and test loss."""
    net = copy.deepcopy(net)
    train_x, train_y, test_x, test_y = ft_data
    inner_opt = optimizer_obj(meta_params, net, lr=inner_lr)
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


class OptimizerTrainer:
    def __init__(
        self,
        optimizer_name,
        task_distribution,
        val_meta_batch_size,
        inner_steps,
        inner_lr,
        train_N,
        meta_lr,
        meta_batch_size,
        noise_std,
        ckpt_path,
    ):
        optimizer_module = importlib.import_module(f"optimizers")
        self.optimizer_obj = getattr(optimizer_module, optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params()

        self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=meta_lr)
        self.task_distribution = task_distribution

        # Inner Loop Hyperparameters
        self.val_meta_batch_size = val_meta_batch_size
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.train_N = train_N
        self.metrics = defaultdict(list)

        # Outer Loop Hyperparameters
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size
        self.noise_std = noise_std

        self.ckpt_path = ckpt_path

    def meta_loss(self):
        """
        Computes the loss of applying a learned optimizer to a given task for some number of steps.
        """
        net = get_pretrained_net(ckpt_path=self.ckpt_path, train=False)
        ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
        _, current_loss = fine_tune(
            self.optimizer_obj,
            net,
            self.meta_params,
            ft_data,
            self.inner_steps,
            self.inner_lr,
        )
        return current_loss

    def meta_train(self):
        """
        Computes the meta-training loss and updates the meta-parameters using evolutionary strategies.
        """
        grads = []
        for _ in range(self.meta_batch_size):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
            epsilon = self.optimizer_obj.get_noise()  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon * self.noise_std
            mp_minus_epsilon = self.meta_params - epsilon * self.noise_std
            _, loss_plus = fine_tune(self.optimizer_obj, net, mp_plus_epsilon, ft_data)
            _, loss_minus = fine_tune(
                self.optimizer_obj, net, mp_minus_epsilon, ft_data
            )
            grads.append((loss_plus - loss_minus) * epsilon / self.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params


def sanity_check(ckpt_path, task_distribution):
    """Sanity check with full vs surgical fine-tuning on only last layer."""
    import optimizers

    meta_params = torch.ones(4).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune(
            optimizers.LayerSGD, net, meta_params, ft_data, inner_steps=10, inner_lr=0.1
        )
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.tensor([-100, -100, -100, 100]).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, loss = fine_tune(
            optimizers.LayerSGD, net, meta_params, ft_data, inner_steps=10, inner_lr=0.1
        )
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")


def run_expt(args):
    pretrain_nets(ckpt_path=args.ckpt_path, num_nets=args.num_nets)
    sanity_check(ckpt_path=args.ckpt_path, task_distribution=args.task_distribution)
    start = time.time()

    opt_trainer = OptimizerTrainer(
        optimizer_name=args.optimizer_name,
        task_distribution=args.task_distribution,
        val_meta_batch_size=args.val_meta_batch_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        train_N=args.train_N,
        meta_lr=args.meta_lr,
        meta_batch_size=args.meta_batch_size,
        noise_std=args.noise_std,
        ckpt_path=args.ckpt_path,
    )

    for meta_step in range(args.meta_steps + 1):
        metrics = defaultdict(list)
        if meta_step % args.val_freq == 0:
            for _ in range(args.val_meta_batch_size):
                current_loss = opt_trainer.meta_loss()
                metrics["loss"].append(current_loss)
            current_loss = np.array(metrics["loss"])
            print(
                f"Meta-step {meta_step}: current loss: {current_loss.mean():.3f}+-{current_loss.std():.3f}"
            )
        meta_params = opt_trainer.meta_train()

    elapsed = time.time() - start
    print(f"Final meta-params: {meta_params}")
    print(f"Time taken: {elapsed:.2f}s")

    # Plot single task
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        task_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
        train_x, train_y, test_x, test_y = task_data
        full_x = torch.tensor(np.linspace(0, 2 * np.pi, 100)).reshape(-1, 1).float()

        full_ft_net, _ = fine_tune(
            opt_trainer.optimizer_obj, net, torch.zeros(4), task_data
        )
        meta_ft_net, _ = fine_tune(
            opt_trainer.optimizer_obj, net, meta_params, task_data
        )
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
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="LayerSGDLinear",
    )
    parser.add_argument("--meta_steps", type=int, default=50)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--val_meta_batch_size", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--train_N", type=int, default=10)
    parser.add_argument(
        "--ckpt_path", type=str, default="/iris/u/yoonho/robust-optimizer/ckpts/sine"
    )
    parser.add_argument("--num_nets", type=int, default=100)
    args = parser.parse_args()
    run_expt(args)