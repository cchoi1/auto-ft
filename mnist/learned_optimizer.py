import copy
import importlib
from collections import defaultdict
from functools import partial

import functorch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mnist import get_dataloaders
from networks import get_pretrained_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def fine_tune(optimizer_obj, inner_steps, inner_lr, _net, meta_params, train_images, train_labels, test_images, test_labels):
    """Fine-tune net on (train_images, train_labels), and return test losses."""
    net = copy.deepcopy(_net)
    inner_opt = optimizer_obj(meta_params, net, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    test_losses = []
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        output = net(test_images)
        test_loss = loss_fn(output, test_labels)
        test_losses.append(test_loss.item())

    return np.array(test_losses)

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


def fine_tune_func_single(optimizer_obj, inner_steps, inner_lr, func_net, net_params, meta_params, train_images, train_labels, test_images, test_labels):
    """Fine-tune the functional model func_net on (train_images, train_labels), and return test losses.
    In the outer loop, we use vmap to parallelize calls to this function for each task in the meta-batch."""
    inner_opt = optimizer_obj(meta_params=meta_params, params=net_params, lr=inner_lr)
    def compute_stateless_loss(params, inputs, labels):
        outputs = func_net(params, inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    test_losses = []
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        gradients = torch.func.grad(compute_stateless_loss)(net_params, train_images, train_labels)
        net_params = inner_opt.update(net_params, gradients)

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = func_net(net_params, test_images)
        test_loss = F.cross_entropy(outputs, test_labels) # (meta_batch_size // 2, 1)
        test_losses.append(test_loss)
    return test_losses


def init_fn(num_nets: int, ckpt_path: str, train: bool):
    """Combines the states of several pretrained nets together by stacking each parameter.
    Returns a stateless version of the model (func_net) and stacked parameters and buffers."""
    nets = []
    for _ in range(num_nets):
        net = copy.deepcopy(get_pretrained_net(ckpt_path=ckpt_path, train=train))
        nets.append(net)
    func_net, batched_weights, buffers = functorch.combine_state_for_ensemble(nets)
    return func_net, batched_weights, buffers


class OptimizerTrainer:
    def __init__(self, args):
        self.ft_distribution = args.ft_distribution
        self.data_dir = args.data_dir
        self.ckpt_path = args.ckpt_path
        self.run_parallel = args.run_parallel
        self.num_workers = args.num_workers
        self.ft_distribution = args.ft_distribution
        self.test_distribution = args.test_distribution
        self.num_nets = args.num_nets

        optimizer_module = importlib.import_module(f"optimizers_func") if self.run_parallel else importlib.import_module(f"optimizers")
        self.optimizer_obj = getattr(optimizer_module, args.optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params()
        self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=args.meta_lr)

        self.train_loader, self.id_val_loader = get_dataloaders(root_dir=args.data_dir, dataset=args.ft_distribution,
                                                                batch_size=args.batch_size,
                                                                meta_batch_size=args.meta_batch_size // 2,
                                                                num_workers=args.num_workers, use_meta_batch=self.run_parallel)
        self.test_loader, self.ood_val_loader = get_dataloaders(root_dir=args.data_dir, dataset=args.test_distribution,
                                                                batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                                num_workers=args.num_workers, use_meta_batch=self.run_parallel)

        # Inner Loop Hyperparameters
        self.val_meta_batch_size = args.val_meta_batch_size
        self.inner_steps = args.inner_steps
        self.inner_lr = args.inner_lr
        self.batch_size = args.batch_size

        # Outer Loop Hyperparameters
        self.meta_lr = args.meta_lr
        self.meta_batch_size = args.meta_batch_size
        self.noise_std = args.noise_std
        self.meta_loss_avg_w = args.meta_loss_avg_w
        self.meta_loss_final_w = args.meta_loss_final_w

        if self.run_parallel and self.num_nets > 1:
            """Use several randomly sampled pretrained models for each finetuning task."""
            _finetune = partial(
                fine_tune_func_n,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
            )
            self.finetune = functorch.vmap(_finetune, in_dims=(None, None, 0, 0, 0, 0, 0, 0))
        elif self.run_parallel and self.num_nets == 1:
            """Use a single pretrained model for all finetuning tasks."""
            self.net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            self.func_net, self.weights = functorch.make_functional(self.net)
            _finetune = partial(
                fine_tune_func_single,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
            )
            self.finetune = functorch.vmap(_finetune, in_dims=(None, None, 0, 0, 0, 0, 0))
            self.finetune_iter = partial(
                fine_tune,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
            )
        else:
            self.finetune_iter = partial(
                fine_tune,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
            )

    def validation_iter(self, repeat):
        losses = defaultdict(list)
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=False)
            train_images, train_labels = next(iter(self.train_loader))
            val_images, val_labels = next(iter(self.ood_val_loader))
            val_losses = self.finetune_iter(net, self.meta_params, train_images, train_labels, val_images,
                                                       val_labels)
            losses["ood"].append(val_losses[-1])
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            train_images, train_labels = next(iter(self.train_loader))
            val_images, val_labels = next(iter(self.id_val_loader))
            train_losses = self.finetune_iter(net, self.meta_params, train_images, train_labels,
                                                           val_images, val_labels)
            losses["id"].append(train_losses[-1])
        id_val_str = f"ID Val loss: {np.mean(losses['id']):.4f} +- {np.std(losses['id']):.4f}"
        ood_val_str = (
            f"OOD Val loss: {np.mean(losses['ood']):.4f} +- {np.std(losses['ood']):.4f}"
        )
        print(id_val_str, '|', ood_val_str)
        return losses

    def validation_parallel(self):
        losses = defaultdict(list)
        meta_params = torch.stack([self.meta_params for _ in range(self.val_meta_batch_size // 2)], dim=0)

        train_loader, id_val_loader = get_dataloaders(root_dir=self.data_dir, dataset=self.ft_distribution,
                                                      batch_size=self.batch_size,
                                                      meta_batch_size=self.val_meta_batch_size // 2,
                                                      num_workers=self.num_workers, use_meta_batch=self.run_parallel)
        test_loader, ood_val_loader = get_dataloaders(root_dir=self.data_dir, dataset=self.test_distribution,
                                                      batch_size=self.batch_size,
                                                      meta_batch_size=self.val_meta_batch_size // 2,
                                                      num_workers=self.num_workers, use_meta_batch=self.run_parallel)

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(ood_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        if self.num_nets > 1:
            func_net, net_params, buffers = init_fn(num_nets=self.val_meta_batch_size // 2, ckpt_path=self.ckpt_path, train=True)
            val_losses = self.finetune(func_net, buffers, net_params, meta_params, train_images, train_labels, val_images,
                                       val_labels)
        else:
            func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
            val_losses = self.finetune(func_net, net_params, meta_params, train_images, train_labels, val_images,
                                       val_labels)

        losses["ood"].append(val_losses[-1].cpu().detach().numpy().mean(axis=0))

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(id_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        if self.num_nets > 1:
            func_net, net_params, params = init_fn(num_nets=self.val_meta_batch_size // 2, ckpt_path=self.ckpt_path, train=False)
            train_losses = self.finetune(func_net, buffers, net_params, meta_params, train_images, train_labels, val_images,
                                       val_labels)
        else:
            func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
            train_losses = self.finetune(func_net, net_params, meta_params, train_images, train_labels, val_images, val_labels)
        losses["id"].append(train_losses[-1].cpu().detach().numpy().mean(axis=0))

        id_val_str = f"ID Val loss: {np.mean(losses['id']):.4f} +- {np.std(losses['id']):.4f}"
        ood_val_str = (
            f"OOD Val loss: {np.mean(losses['ood']):.4f} +- {np.std(losses['ood']):.4f}"
        )
        print(id_val_str, '|', ood_val_str)
        return losses

    def validation(self, repeat):
        if self.run_parallel:
            return self.validation_parallel()
        else:
            return self.validation_iter(repeat)

    def outer_loop_step_iter(self, net=None):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling."""
        grads = []
        for _ in range(self.meta_batch_size // 2):
            if net is None:
                net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            epsilon = (
                    self.optimizer_obj.get_noise() * self.noise_std
            )  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon
            mp_minus_epsilon = self.meta_params - epsilon
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.ood_val_loader))
            losses_plus = self.finetune_iter(net, mp_plus_epsilon, train_images, train_labels, test_images,
                                                 test_labels)
            losses_minus = self.finetune_iter(net, mp_minus_epsilon, train_images, train_labels, test_images,
                                                  test_labels)
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

    def outer_loop_step_parallel(self):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
        Parallelizes over tasks in the meta batch using vmap."""
        epsilon = (
                self.optimizer_obj.get_noise(self.meta_batch_size // 2) * self.noise_std
        )  # Antithetic sampling
        mp_plus_epsilon = self.meta_params + epsilon
        mp_minus_epsilon = self.meta_params - epsilon

        # Prepare a meta-batch of fine-tuning tasks
        train_images, train_labels = next(iter(self.train_loader))
        test_images, test_labels = next(iter(self.test_loader))
        train_images = train_images.view(self.meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.meta_batch_size // 2, self.batch_size)
        test_images = test_images.view(self.meta_batch_size // 2, self.batch_size, 28, 28)
        test_labels = test_labels.view(self.meta_batch_size // 2, self.batch_size)

        if self.num_nets == 1:
            func_net, weights = functorch.make_functional(copy.deepcopy(self.net))
            losses_plus = self.finetune(func_net, weights, mp_plus_epsilon, train_images, train_labels, test_images, test_labels)
            losses_minus = self.finetune(func_net, weights, mp_minus_epsilon, train_images, train_labels, test_images, test_labels)
        else:
            func_net, net_params, buffers = init_fn(num_nets=self.meta_batch_size // 2, ckpt_path=self.ckpt_path, train=True)
            losses_plus = self.finetune(func_net, buffers, net_params, mp_plus_epsilon, train_images, train_labels,
                                        test_images, test_labels)
            losses_minus = self.finetune(func_net, buffers, net_params, mp_minus_epsilon, train_images, train_labels,
                                         test_images, test_labels)
        losses_plus = torch.stack(losses_plus, dim=0) # (meta_batch_size//2, inner_steps)
        losses_minus = torch.stack(losses_minus, dim=0) # (meta_batch_size//2, inner_steps)
        loss_diff = losses_plus - losses_minus # (meta_batch_size//2, inner_steps)

        objective = (
                loss_diff[:, -1].mean() * self.meta_loss_final_w
                + loss_diff.mean() * self.meta_loss_avg_w
        )
        grad = objective * epsilon / self.noise_std / 2
        grads_mean = grad.mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params

    def outer_loop_step(self):
        if self.run_parallel:
            return self.outer_loop_step_parallel()
        else:
            return self.outer_loop_step_iter()