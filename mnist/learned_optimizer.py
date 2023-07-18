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
from networks import get_pretrained_net_fixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune(optimizer_obj, inner_steps, inner_lr, features, _net, meta_params, train_images, train_labels, test_images, test_labels, iter):
    """Fine-tune net on (train_images, train_labels), and return test losses."""
    net = copy.deepcopy(_net)
    inner_opt = optimizer_obj(meta_params, net, features, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    test_losses = []
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step(curr_loss=loss.item(), iter=iter)

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        output = net(test_images)
        test_loss = loss_fn(output, test_labels)
        test_losses.append(test_loss.item())

    return np.array(test_losses)

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
        test_loss = compute_stateless_loss(net_params, test_images, test_labels)
        test_losses.append(test_loss)

    return test_losses


class OptimizerTrainer:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.ckpt_path = args.ckpt_path
        self.run_parallel = args.run_parallel
        self.num_workers = args.num_workers
        self.ft_id_dist = args.ft_id_dist
        self.ft_ood_dist = args.ft_ood_dist
        self.test_dist = args.test_dist
        self.num_nets = args.num_nets
        self.features = args.features
        self.net = get_pretrained_net_fixed(ckpt_path=self.ckpt_path, dataset_name= args.pretrain_dist, train=True)
        if self.features is not None:
            self.num_features = len(self.features)
        else:
            self.num_features = len([p for p in self.net.parameters()])
        self.num_iters = 0

        optimizer_module = importlib.import_module(f"optimizers_func") if self.run_parallel else importlib.import_module(f"optimizers")
        self.optimizer_obj = getattr(optimizer_module, args.optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params(self.num_features)
        self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=args.meta_lr)

        _, self.source_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=["mnist"], batch_size=args.batch_size,
                                                    meta_batch_size=args.meta_batch_size // 2,
                                                    num_workers=args.num_workers, use_meta_batch=self.run_parallel)
        self.train_loader, self.id_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_id_dist],
                                                                batch_size=args.batch_size,
                                                                meta_batch_size=args.meta_batch_size // 2,
                                                                num_workers=args.num_workers, use_meta_batch=self.run_parallel)
        self.ood_val1_loader, self.ood_val2_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
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

        if self.run_parallel:
            """Use a single pretrained model for all finetuning tasks."""
            # self.func_net, self.weights = functorch.make_functional(self.net)
            _finetune = partial(
                fine_tune_func_single,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
            )
            self.finetune = functorch.vmap(_finetune, in_dims=(None, None, 0, 0, 0, 0, 0))
        else:
            self.finetune_iter = partial(
                fine_tune,
                self.optimizer_obj,
                self.inner_steps,
                self.inner_lr,
                self.features
            )

    def validation_iter(self, repeat):
        losses = defaultdict(list)

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            source_val_images, source_val_labels = next(iter(self.id_val_loader))
            source_val_losses = self.finetune_iter(net, self.meta_params, train_images, train_labels, source_val_images, source_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            losses["source"].append(source_val_losses[-1])

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            id_val_images, id_val_labels = next(iter(self.id_val_loader))
            id_val_losses = self.finetune_iter(net, self.meta_params, train_images, train_labels, id_val_images, id_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            losses["id"].append(id_val_losses[-1])

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            ood_val_images, ood_val_labels = next(iter(self.ood_val2_loader))
            ood_val_losses = self.finetune_iter(net, self.meta_params, train_images, train_labels, ood_val_images, ood_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            losses["ood"].append(ood_val_losses[-1])

        source_val_str = f"Source Val loss: {np.mean(losses['source']):.4f} +- {np.std(losses['source']):.4f}"
        id_val_str = f"ID Val loss: {np.mean(losses['id']):.4f} +- {np.std(losses['id']):.4f}"
        ood_val_str = (
            f"OOD Val loss: {np.mean(losses['ood']):.4f} +- {np.std(losses['ood']):.4f}"
        )
        print(source_val_str, '|', id_val_str, '|', ood_val_str)
        return losses

    def validation_parallel(self):
        losses = defaultdict(list)
        meta_params = torch.stack([self.meta_params for _ in range(self.val_meta_batch_size // 2)], dim=0)

        train_loader, id_val_loader = get_dataloaders(root_dir=self.data_dir, dataset_names=[self.ft_id_dist],
                                                      batch_size=self.batch_size,
                                                      meta_batch_size=self.val_meta_batch_size // 2,
                                                      num_workers=self.num_workers, use_meta_batch=self.run_parallel)
        test_loader, ood_val_loader = get_dataloaders(root_dir=self.data_dir, dataset_names=[self.ft_ood_dist],
                                                      batch_size=self.batch_size,
                                                      meta_batch_size=self.val_meta_batch_size // 2,
                                                      num_workers=self.num_workers, use_meta_batch=self.run_parallel)

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(ood_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
        val_losses = self.finetune(func_net, net_params, meta_params, train_images, train_labels, val_images, val_labels)

        losses["ood"].append(val_losses[-1].cpu().detach().numpy().mean(axis=0))

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(id_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
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

    def outer_loop_step_iter(self, _net=None, epsilon=None, train_x=None, train_y=None, test_x=None, test_y=None):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
        Only pass in params _net, epsilon, train_x, train_y, test_x, test_y when unit-testing."""
        grads = []
        all_losses_diff = []
        for _ in range(self.meta_batch_size // 2):
            net = copy.deepcopy(self.net)
            if epsilon is None:
                epsilon = (
                        self.optimizer_obj.get_noise(self.num_features) * self.noise_std
                ) # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon
            mp_minus_epsilon = self.meta_params - epsilon
            if train_x is None or train_y is None or test_x is None or test_y is None:
                train_images, train_labels = next(iter(self.train_loader))
                test_images, test_labels = next(iter(self.ood_val1_loader))
            else:
                train_images, train_labels = train_x[_:(_+1)].squeeze(0), train_y[_:(_+1)].squeeze(0)
                test_images, test_labels = test_x[_:(_+1)].squeeze(0), test_y[_:(_+1)].squeeze(0)

            losses_plus = self.finetune_iter(net, mp_plus_epsilon, train_images, train_labels, test_images, test_labels, self.num_iters)
            self.num_iters += self.inner_steps
            losses_minus = self.finetune_iter(net, mp_minus_epsilon, train_images, train_labels, test_images, test_labels, self.num_iters)
            self.num_iters += self.inner_steps

            loss_diff = losses_plus - losses_minus
            all_losses_diff.append(loss_diff)
            objective = (
                    loss_diff[-1] * self.meta_loss_final_w
                    + loss_diff.mean() * self.meta_loss_avg_w
            )
            grads.append(objective * epsilon / self.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params, grads_mean

    def outer_loop_step_parallel(self, net=None, epsilon=None, train_x=None, train_y=None, test_x=None, test_y=None):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
        Parallelizes over tasks in the meta batch using vmap.
        Only pass in params _net, epsilon, train_x, train_y, test_x, test_y when unit-testing."""
        if epsilon is None:
            epsilon = (
                    self.optimizer_obj.get_noise(self.meta_batch_size // 2) * self.noise_std
            ) # Antithetic sampling
        mp_plus_epsilon = self.meta_params + epsilon
        mp_minus_epsilon = self.meta_params - epsilon

        # Prepare a meta-batch of fine-tuning tasks
        if train_x is None or train_y is None or test_x is None or test_y is None:
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.ood_val1_loader))
        else:
            train_images, train_labels, test_images, test_labels = train_x, train_y, test_x, test_y
        x_shape = (self.meta_batch_size // 2, self.batch_size, 28, 28)
        y_shape = (self.meta_batch_size // 2, self.batch_size)
        train_images, test_images = train_images.view(x_shape), test_images.view(x_shape)
        train_labels, test_labels = train_labels.view(y_shape), test_labels.view(y_shape)

        net_copy = copy.deepcopy(self.net)
        func_net, net_params = functorch.make_functional(net_copy)
        losses_plus = self.finetune(func_net, net_params, mp_plus_epsilon, train_images, train_labels, test_images, test_labels)
        losses_minus = self.finetune(func_net, net_params, mp_minus_epsilon, train_images, train_labels, test_images, test_labels)
        losses_plus = torch.stack(losses_plus, dim=0).transpose(0, 1)
        losses_minus = torch.stack(losses_minus, dim=0).transpose(0, 1) 
        loss_diff = losses_plus - losses_minus

        objective = (
                loss_diff[:, -1].mean() * self.meta_loss_final_w
                + loss_diff.mean() * self.meta_loss_avg_w
        )
        grad = objective * epsilon / self.noise_std / 2
        grads_mean = grad.mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params, grads_mean

    def outer_loop_step(self):
        if self.run_parallel:
            return self.outer_loop_step_parallel()
        else:
            return self.outer_loop_step_iter()
    