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
from mnist import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def fine_tune(optimizer_name, net, meta_params, train_loader, test_loader, inner_steps=10, inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net and test loss."""
    net = copy.deepcopy(net)
    if optimizer_name == "LayerSGD":
        inner_opt = LayerSGD(meta_params, net.parameters(), lr=inner_lr)
    elif optimizer_name == "LayerSGDLinear":
        inner_opt = LayerSGDLinear(meta_params, net, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    test_losses = []
    for _ in range(inner_steps):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)
            loss = loss_fn(preds, labels)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_preds = net(images)
            test_loss = loss_fn(test_preds, labels)
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
        start_time = time.time()
        net = get_pretrained_net(ckpt_path=self.ckpt_path, train=False)
        train_loader, test_loader = load_mnist(dataset=self.task_distribution)
        _, current_loss = fine_tune(self.optimizer_name, net, self.meta_params, train_loader, test_loader, self.inner_steps, self.inner_lr)
        end_time = time.time()
        print("Inner Loop| Time: {:.2f}".format(end_time - start_time))
        return current_loss

    def meta_train(self):
        """
        Computes the meta-training loss and updates the meta-parameters using evolutionary strategies.
        """
        grads = []
        for i in range(self.meta_batch_size):
            start_time = time.time()
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            train_loader, test_loader = load_mnist(dataset=self.task_distribution)
            epsilon = torch.randn(size=self.meta_params.shape)  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon * self.noise_std
            mp_minus_epsilon = self.meta_params - epsilon * self.noise_std
            _, loss_plus = fine_tune(self.optimizer_name, net, mp_plus_epsilon, train_loader, test_loader, self.inner_steps, self.inner_lr)
            _, loss_minus = fine_tune(self.optimizer_name, net, mp_minus_epsilon, train_loader, test_loader, self.inner_steps, self.inner_lr)
            grads.append((loss_plus - loss_minus) * epsilon / self.noise_std / 2)
        end_time = time.time()
        print("Outer Loop Step: {}/{} | Time: {:.2f}".format(i, self.meta_batch_size, end_time - start_time))
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
        # metrics = defaultdict(list)
        # if meta_step % args.val_freq == 0:
        #     for _ in range(args.val_meta_batch_size):
        #         current_loss = opt_trainer.meta_loss()
        #         metrics["loss"].append(current_loss)
        #     current_loss = np.array(metrics["loss"])
        #     print(
        #         f"Meta-step {meta_step}: current loss: {current_loss.mean():.3f}+-{current_loss.std():.3f}"
        #     )
        meta_params = opt_trainer.meta_train()
    print(f"Final meta-params: {meta_params}")
    return meta_params