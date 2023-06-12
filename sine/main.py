import argparse
import copy
import importlib
import os
import time
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from networks import get_pretrained_net, pretrain_nets
from tasks import sample_ft_data, savefig


def fine_tune(_net, meta_params, ft_data, optimizer_obj, inner_steps=10, inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net and test loss."""
    net = copy.deepcopy(_net)
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
    return net, np.array(test_losses)


class OptimizerTrainer:
    def __init__(self, args):
        optimizer_module = importlib.import_module(f"optimizers")
        self.optimizer_obj = getattr(optimizer_module, args.optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params()

        self.task_distribution = args.task_distribution
        self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=args.meta_lr)

        # Inner Loop Hyperparameters
        self.val_meta_batch_size = args.val_meta_batch_size
        self.inner_steps = args.inner_steps
        self.inner_lr = args.inner_lr
        self.train_N = args.train_N

        # Outer Loop Hyperparameters
        self.meta_lr = args.meta_lr
        self.meta_batch_size = args.meta_batch_size
        self.noise_std = args.noise_std
        self.meta_loss_avg_w = args.meta_loss_avg_w
        self.meta_loss_final_w = args.meta_loss_final_w

        self.ckpt_path = args.ckpt_path

        self.finetune = partial(
            fine_tune,
            optimizer_obj=self.optimizer_obj,
            inner_steps=self.inner_steps,
            inner_lr=self.inner_lr,
        )

    def validation(self, repeat):
        losses = defaultdict(list)
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=False)
            ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
            _, val_losses = self.finetune(net, self.meta_params, ft_data)
            val_loss = val_losses[-1]
            losses["val"].append(val_loss)
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
            _, train_losses = self.finetune(net, self.meta_params, ft_data)
            train_loss = train_losses[-1]
            losses["train"].append(train_loss)
        train_str = f"Train loss: {np.mean(losses['train']):.4f} +- {np.std(losses['train']):.4f}"
        val_str = (
            f"Val loss: {np.mean(losses['val']):.4f} +- {np.std(losses['val']):.4f}"
        )
        print(train_str, val_str)
        return losses

    def outer_loop_step(self):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling."""
        grads = []
        for _ in range(self.meta_batch_size // 2):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
            epsilon = (
                self.optimizer_obj.get_noise() * self.noise_std
            )  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon
            mp_minus_epsilon = self.meta_params - epsilon
            _, losses_plus = self.finetune(net, mp_plus_epsilon, ft_data)
            _, losses_minus = self.finetune(net, mp_minus_epsilon, ft_data)
            loss_diff = losses_plus - losses_minus
            objective = (
                loss_diff[-1] * self.meta_loss_final_w
                + loss_diff.mean() * self.meta_loss_avg_w
            )
            grads.append(objective * epsilon / self.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params


def sanity_check(ckpt_path, task_distribution):
    """Sanity check with full vs surgical fine-tuning on only last layer."""
    import optimizers

    inner_steps = 3

    meta_params = torch.ones(4).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, ft_losses = fine_tune(
            net,
            meta_params,
            ft_data,
            optimizers.LayerSGD,
            inner_steps=inner_steps,
            inner_lr=0.1,
        )
        loss = ft_losses[-1]
        losses.append(loss)
    losses = np.array(losses)
    print(f"Default: {losses.mean():.4f} +- {losses.std():.4f}")

    meta_params = torch.tensor([-100, -100, -100, 100]).float()
    losses = []
    for _ in range(100):
        net = get_pretrained_net(ckpt_path=ckpt_path, train=False)
        ft_data = sample_ft_data(task_distribution)
        _, ft_losses = fine_tune(
            net,
            meta_params,
            ft_data,
            optimizers.LayerSGD,
            inner_steps=inner_steps,
            inner_lr=0.1,
        )
        loss = ft_losses[-1]
        losses.append(loss)
    losses = np.array(losses)
    print(f"Last bias only: {losses.mean():.4f} +- {losses.std():.4f}")


def plot_learned(args, optimizer_obj, meta_params):
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        task_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
        train_x, train_y, test_x, test_y = task_data
        full_x = torch.tensor(np.linspace(0, 2 * np.pi, 100)).reshape(-1, 1).float()

        initial_preds = net(full_x).detach()
        ax.scatter(train_x, train_y, c="k", label="train", zorder=10, s=50)
        ax.plot(full_x, initial_preds, c="gray", label="pretrained")

        meta_ft_net, _ = fine_tune(
            net, meta_params, task_data, optimizer_obj, args.inner_steps, args.inner_lr
        )
        meta_ft_preds = meta_ft_net(full_x).detach()
        ax.plot(full_x, meta_ft_preds, c="blue", label="meta+ft")

        if args.optimizer_name in ["LayerSGD", "LayerSGDLiner"]:
            full_ft_net, _ = fine_tune(
                net,
                torch.zeros(4),
                task_data,
                optimizer_obj,
                args.inner_steps,
                args.inner_lr,
            )
            full_ft_preds = full_ft_net(full_x).detach()
            ax.plot(full_x, full_ft_preds, c="red", label="full ft")
        ax.scatter(test_x, test_y, c="gray", alpha=0.3)
        min_y, max_y = test_y.min(), test_y.max()
        ax.set_ylim(min(-2.0, min_y - 0.1), max(2.0, max_y + 0.1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    savefig(f"{args.task_distribution}_meta_ft.png")


def plot_learning_curves(args, optimizer_obj, meta_params):
    loss_fn = nn.MSELoss()
    all_losses = defaultdict(list)
    for _ in tqdm(range(100)):
        net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        task_data = sample_ft_data(args.task_distribution, train_N=args.train_N)
        train_x, train_y, test_x, test_y = task_data
        inner_opt = optimizer_obj(meta_params, net, lr=1e-1)
        losses = defaultdict(list)
        test_steps = max(10, args.inner_steps * 2)
        for _ in range(test_steps):
            test_preds = net(test_x)
            test_loss = loss_fn(test_preds, test_y)

            preds = net(train_x)
            train_loss = loss_fn(preds, train_y)
            inner_opt.zero_grad()
            train_loss.backward()
            inner_opt.step()

            losses["train"].append(train_loss.item())
            losses["test"].append(test_loss.item())
        all_losses["train"].append(np.array(losses["train"]))
        all_losses["test"].append(np.array(losses["test"]))

    plt.figure(figsize=(4, 2.5))
    for test_traj in all_losses["test"]:
        plt.plot(test_traj, c="gray", alpha=0.3)
    avg_train_loss = np.stack(all_losses["train"]).mean(axis=0)
    avg_test_loss = np.stack(all_losses["test"]).mean(axis=0)
    plt.plot(avg_train_loss, "o--", label="train")
    plt.plot(avg_test_loss, "o-", label="test")
    plt.axvline(args.inner_steps, c="k", ls="--")
    plt.legend()
    plt.yscale("log")
    savefig(f"{args.task_distribution}_learning_curves.png")


def run_expt(args):
    pretrain_nets(ckpt_path=args.ckpt_path, num_nets=args.num_nets)
    sanity_check(ckpt_path=args.ckpt_path, task_distribution=args.task_distribution)
    start = time.time()

    opt_trainer = OptimizerTrainer(args)

    for meta_step in range(args.meta_steps + 1):
        if meta_step % args.val_freq == 0:
            opt_trainer.validation(repeat=args.val_meta_batch_size)
        meta_params = opt_trainer.outer_loop_step()

    elapsed = time.time() - start
    print(f"Final meta-params: {meta_params}")
    print(f"Time taken: {elapsed:.2f}s")

    plot_learned(args, opt_trainer.optimizer_obj, meta_params)
    plot_learning_curves(args, opt_trainer.optimizer_obj, meta_params)

    return meta_params


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
    parser.add_argument("--meta_steps", type=int, default=100)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--meta_loss_avg_w", type=float, default=0.0)
    parser.add_argument("--meta_loss_final_w", type=float, default=1.0)
    parser.add_argument("--meta_batch_size", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--val_meta_batch_size", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--train_N", type=int, default=10)
    parser.add_argument(
        "--ckpt_path", type=str, default="/iris/u/yoonho/robust-optimizer/ckpts/sine"
    )
    parser.add_argument("--num_nets", type=int, default=100)
    args = parser.parse_args()

    defaults = vars(parser.parse_args([]))
    actuals = vars(args)
    nondefaults = []
    for k in sorted(actuals.keys()):
        if defaults[k] != actuals[k]:
            first_letters = "".join([w[0] for w in k.split("_")])
            if type(actuals[k]) == float:
                nondefaults.append(f"{first_letters}={actuals[k]:.2e}")
            else:
                nondefaults.append(f"{first_letters}={actuals[k]}")
    if len(nondefaults) > 0:
        args.exp_name = "_".join(nondefaults)
    else:
        args.exp_name = "default"

    meta_params = run_expt(args)

    os.makedirs("results", exist_ok=True)
    np.save(f"results/{args.exp_name}.npy", meta_params)
    print(f"Saved results to results/{args.exp_name}.npy")
