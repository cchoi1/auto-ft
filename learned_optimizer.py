#%%
import argparse
import copy
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from optimizers import LayerSGD, LayerSGDLinear
from models import get_pretrained_net
from sine import generate_sine_data

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

class OptimizerTrainer:
    def __init__(
            self,
            optimizer_name,
            net,
            task_distribution,
            val_meta_batch_size,
            inner_steps,
            inner_lr,
            train_N,
            meta_lr,
            meta_batch_size,
            noise_std,
            ckpt_path
    ):
        self.optimizer_name = optimizer_name
        if optimizer_name == "LayerSGD":
            self.meta_params = LayerSGD.get_init_meta_params()
            self.optimizer = LayerSGD(self.meta_params, net.parameters(), lr=inner_lr)
        elif optimizer_name == "LayerSGDLinear":
            self.meta_params = LayerSGDLinear.get_init_meta_params()
            self.optimizer = LayerSGDLinear(self.meta_params, net, lr=inner_lr)
        else:
            raise ValueError(f"Unknown optimizer name {self.optimizer_name}")
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
        _, current_loss = fine_tune(self.optimizer_name, net, self.meta_params, ft_data, self.inner_steps, self.inner_lr)
        return current_loss

    def meta_train(self):
        """
        Computes the meta-training loss and updates the meta-parameters using evolutionary strategies.
        """
        grads = []
        for _ in range(self.meta_batch_size):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            ft_data = sample_ft_data(self.task_distribution, train_N=self.train_N)
            epsilon = torch.randn(size=self.meta_params.shape)  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon * self.noise_std
            mp_minus_epsilon = self.meta_params - epsilon * self.noise_std
            _, loss_plus = fine_tune(self.optimizer_name, net, mp_plus_epsilon, ft_data)
            _, loss_minus = fine_tune(self.optimizer_name, net, mp_minus_epsilon, ft_data)
            grads.append((loss_plus - loss_minus) * epsilon / self.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params

def metalearn_opt(args):
    net = get_pretrained_net(ckpt_path=args.ckpt_path, train=True)
    opt_trainer = OptimizerTrainer(
        optimizer_name=args.optimizer_name,
        net=net,
        task_distribution=args.task_distribution,
        val_meta_batch_size=args.val_meta_batch_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        train_N=args.train_N,
        meta_lr=args.meta_lr,
        meta_batch_size=args.meta_batch_size,
        noise_std=args.noise_std,
        ckpt_path=args.ckpt_path
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
    print(f"Final meta-params: {meta_params}")
    return meta_params