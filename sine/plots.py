#%%
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import optimizers
from main import fine_tune
from networks import get_pretrained_net
from tasks import sample_ft_data, savefig

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


# %%
exp_dir = "results/ms=1000_on=LOptNet_td=amp_shift"
exp_dir = "results/ms=200_on=LOptNet"
args = pickle.load(open(f"{exp_dir}/args.pkl", "rb"))
print(args)
optimizer_obj = getattr(optimizers, args.optimizer_name)
meta_params = np.load(f"{exp_dir}/final.npy")
meta_params = torch.tensor(meta_params).float()
plot_learned(args, optimizer_obj, meta_params)
plot_learning_curves(args, optimizer_obj, meta_params)

# %%
