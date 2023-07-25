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
from utils import get_lopt_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune(optimizer_obj, inner_lr, inp_info, total_iters, _net, meta_params, inner_steps, train_images, train_labels, test_images, test_labels, iter):
    """Fine-tune net on (train_images, train_labels), and return test losses."""
    net = copy.deepcopy(_net)
    inner_opt = optimizer_obj(meta_params, net, inp_info, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    test_losses = []
    test_accs = []
    correct = 0.0; total = 0.0
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step(curr_loss=loss.item(), iter=iter, iter_frac=iter/total_iters)

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        output = net(test_images)
        test_loss = loss_fn(output, test_labels)
        test_losses.append(test_loss.item())
        preds = output.argmax(dim=1)
        correct += (preds == test_labels).sum().item()
        total += test_labels.shape[0]
        test_accs.append(correct / total)

    return np.array(test_losses), np.array(test_accs)

def fine_tune_func_single(optimizer_obj, inner_lr, func_net, net_params, meta_params, inner_steps, train_images, train_labels, test_images, test_labels):
    """Fine-tune the functional model func_net on (train_images, train_labels), and return test losses.
    In the outer loop, we use vmap to parallelize calls to this function for each task in the meta-batch."""
    inner_opt = optimizer_obj(meta_params=meta_params, params=net_params, lr=inner_lr)
    def compute_stateless_loss(params, inputs, labels):
        outputs = func_net(params, inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    test_losses = []
    test_accs = []
    correct = 0.0; total = 0.0
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        gradients = torch.func.grad(compute_stateless_loss)(net_params, train_images, train_labels)
        net_params = inner_opt.update(net_params, gradients)

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_loss = compute_stateless_loss(net_params, test_images, test_labels)
        test_losses.append(test_loss)
        preds = torch.argmax(func_net(net_params, test_images), dim=-1)
        correct += (preds == test_labels).sum().item()
        total += test_labels.shape[0]
        test_accs.append(correct / total)

    return test_losses, test_accs


class OptimizerTrainer:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.ckpt_path = args.ckpt_path
        self.run_parallel = args.run_parallel
        self.num_workers = args.num_workers
        self.pretrain_dist = args.pretrain_dist
        self.ft_id_dist = args.ft_id_dist
        self.ft_ood_dist = args.ft_ood_dist
        self.test_dist = args.test_dist
        self.num_nets = args.num_nets
        self.features = args.features
        self.net = get_pretrained_net_fixed(ckpt_path=self.ckpt_path, dataset_name=args.pretrain_dist, train=True)
        self.lopt_info = get_lopt_info(self.features, self.net)
        self.num_iters = 0

        optimizer_module = importlib.import_module(f"optimizers.optimizers_func") if self.run_parallel else importlib.import_module(f"optimizers.optimizers")
        self.optimizer_obj = getattr(optimizer_module, args.optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params(self.lopt_info)
        if type(self.meta_params) == list:
            self.meta_optimizer = torch.optim.SGD(self.meta_params, lr=args.meta_lr)
        else:
            self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=args.meta_lr)

        self.train_loader, self.id_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_id_dist],
                                                                batch_size=args.batch_size,
                                                                meta_batch_size=args.meta_batch_size // 2,
                                                                num_workers=args.num_workers, use_meta_batch=self.run_parallel)
        self.ood_val1_loader, self.ood_val2_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
                                                                batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                                num_workers=args.num_workers, use_meta_batch=self.run_parallel)
        _, self.test_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.test_dist],
                                              batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                              num_workers=args.num_workers, use_meta_batch=self.run_parallel)

        # Inner Loop Hyperparameters
        self.val_meta_batch_size = args.val_meta_batch_size
        self.val_inner_steps = args.val_inner_steps
        self.inner_steps = args.inner_steps
        self.inner_lr = args.inner_lr
        self.batch_size = args.batch_size

        # Outer Loop Hyperparameters
        self.meta_lr = args.meta_lr
        self.meta_batch_size = args.meta_batch_size
        self.noise_std = args.noise_std
        self.meta_loss_avg_w = args.meta_loss_avg_w
        self.meta_loss_final_w = args.meta_loss_final_w
        self.meta_steps = args.meta_steps
        self.total_iters = self.meta_steps * self.inner_steps

        if self.run_parallel:
            """Use a single pretrained model for all finetuning tasks."""
            # self.func_net, self.weights = functorch.make_functional(self.net)
            _finetune = partial(
                fine_tune_func_single,
                self.optimizer_obj,
                self.inner_lr,
            )
            self.finetune = functorch.vmap(_finetune, in_dims=(None, None, 0, 0, 0, 0, 0))
        else:
            self.finetune_iter = partial(
                fine_tune,
                self.optimizer_obj,
                self.inner_lr,
                self.lopt_info,
                self.total_iters,
            )

    def validation_iter(self, repeat):
        metrics = defaultdict(list)

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            source_val_images, source_val_labels = next(iter(self.train_loader))
            source_val_losses, source_val_accs = self.finetune_iter(net, self.meta_params, self.val_inner_steps, train_images, train_labels, source_val_images, source_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            metrics["source_loss"].append(source_val_losses[-1])
            metrics["source_acc"].append(source_val_accs[-1])

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            id_val_images, id_val_labels = next(iter(self.id_val_loader))
            id_val_losses, id_val_accs = self.finetune_iter(net, self.meta_params, self.val_inner_steps, train_images, train_labels, id_val_images, id_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            metrics["id_loss"].append(id_val_losses[-1])
            metrics["id_acc"].append(id_val_accs[-1])

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            ood_val_images, ood_val_labels = next(iter(self.ood_val2_loader))
            ood_val_losses, ood_val_accs = self.finetune_iter(net, self.meta_params, self.val_inner_steps, train_images, train_labels, ood_val_images, ood_val_labels, self.num_iters)
            self.num_iters += self.inner_steps
            metrics["ood_loss"].append(ood_val_losses[-1])
            metrics["ood_acc"].append(ood_val_accs[-1])

        net = copy.deepcopy(self.net)
        for _ in range(repeat):
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.test_loader))
            test_losses, test_accs = self.finetune_iter(net, self.meta_params, self.val_inner_steps, train_images, train_labels, test_images, test_labels, self.num_iters)
            self.num_iters += self.inner_steps
            metrics["test_loss"].append(test_losses[-1])
            metrics["test_acc"].append(test_accs[-1])

        source_val_str = f"Source Loss: {np.mean(metrics['source_loss']):.4f} +- {np.std(metrics['source_loss']):.4f} " \
                         f"Source Acc: {100 * np.mean(metrics['source_acc']):.4f} +- {100 * np.std(metrics['source_acc']):.4f}"
        id_val_str = f"\nID Loss: {np.mean(metrics['id_loss']):.4f} +- {np.std(metrics['id_loss']):.4f} " \
                     f"ID Acc: {100 * np.mean(metrics['id_acc']):.4f} +- {100 * np.std(metrics['id_acc']):.4f}"
        ood_val_str = (
            f"\nOOD Loss: {np.mean(metrics['ood_loss']):.4f} +- {np.std(metrics['ood_loss']):.4f} "
            f"OOD Acc: {100 * np.mean(metrics['ood_acc']):.4f} +- {100 * np.std(metrics['ood_acc']):.4f}"
        )
        test_str = (
            f"\nTest Loss: {np.mean(metrics['test_loss']):.4f} +- {np.std(metrics['test_loss']):.4f} "
            f"Test Acc: {100 * np.mean(metrics['test_acc']):.4f} +- {100 * np.std(metrics['test_acc']):.4f}"
        )
        print(source_val_str, '|', id_val_str, '|', ood_val_str, '|', test_str, '\n')
        return metrics

    def validation_parallel(self):
        metrics = defaultdict(list)
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
        train_images2, train_labels2 = next(iter(train_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        train_images2 = train_images2.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels2 = train_labels2.view(self.val_meta_batch_size // 2, self.batch_size)
        func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
        train_losses, train_accs = self.finetune(func_net, net_params, meta_params, self.val_inner_steps, train_images, train_labels, train_images2, train_labels2)
        metrics["source_loss"].append(train_losses[-1].cpu().detach().numpy().mean(axis=0))
        metrics["source_acc"].append(train_accs[-1].cpu().detach().numpy().mean(axis=0))

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(id_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
        id_val_losses, id_val_accs = self.finetune(func_net, net_params, meta_params, self.val_inner_steps, train_images, train_labels, val_images, val_labels)
        metrics["id_loss"].append(id_val_losses[-1].cpu().detach().numpy().mean(axis=0))
        metrics["id_acc"].append(id_val_accs[-1].cpu().detach().numpy().mean(axis=0))

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(ood_val_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        val_images = val_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        val_labels = val_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
        ood_val_losses, ood_val_accs = self.finetune(func_net, net_params, meta_params, self.val_inner_steps, train_images, train_labels, val_images, val_labels)
        metrics["ood_loss"].append(ood_val_losses[-1].cpu().detach().numpy().mean(axis=0))
        metrics["ood_acc"].append(ood_val_accs[-1].cpu().detach().numpy().mean(axis=0))

        train_images, train_labels = next(iter(train_loader))
        test_images, test_labels = next(iter(test_loader))
        train_images = train_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        train_labels = train_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        test_images = test_images.view(self.val_meta_batch_size // 2, self.batch_size, 28, 28)
        test_labels = test_labels.view(self.val_meta_batch_size // 2, self.batch_size)
        func_net, net_params = functorch.make_functional(copy.deepcopy(self.net))
        test_losses, test_accs = self.finetune(func_net, net_params, meta_params, self.val_inner_steps, train_images, train_labels, test_images, test_labels)
        metrics["test_loss"].append(test_losses[-1].cpu().detach().numpy().mean(axis=0))
        metrics["test_acc"].append(test_accs[-1].cpu().detach().numpy().mean(axis=0))

        source_val_str = f"\nSource Loss: {np.mean(metrics['source_loss']):.4f} +- {np.std(metrics['source_loss']):.4f} " \
                         f"Source Acc: {100 * np.mean(metrics['source_acc']):.4f} +- {100 * np.std(metrics['source_acc']):.4f}"
        id_val_str = f"\nID Loss: {np.mean(metrics['id_loss']):.4f} +- {np.std(metrics['id_loss']):.4f} " \
                     f"ID Acc: {100 * np.mean(metrics['id_acc']):.4f} +- {100 * np.std(metrics['id_acc']):.4f}"
        ood_val_str = (
            f"\nOOD Loss: {np.mean(metrics['ood_loss']):.4f} +- {np.std(metrics['ood_loss']):.4f} "
            f"OOD Acc: {100 * np.mean(metrics['ood_acc']):.4f} +- {100 * np.std(metrics['ood_acc']):.4f}"
        )
        test_str = (
            f"\nTest Loss: {np.mean(metrics['test_loss']):.4f} +- {np.std(metrics['test_loss']):.4f} "
            f"Test Acc: {100 * np.mean(metrics['test_acc']):.4f} +- {100 * np.std(metrics['test_acc']):.4f}"
        )
        print(source_val_str, '|', id_val_str, '|', ood_val_str, '|', test_str, '\n')
        return metrics

    def validation(self, repeat):
        if self.run_parallel:
            return self.validation_parallel()
        else:
            return self.validation_iter(repeat)

    def outer_loop_step_iter(self, _net=None, epsilon=None, train_x=None, train_y=None, test_x=None, test_y=None):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
        Only pass in params _net, epsilon, train_x, train_y, test_x, test_y when unit-testing."""
        grads = []
        for _ in range(self.meta_batch_size // 2):
            net = copy.deepcopy(self.net)
            epsilon = (
                    self.optimizer_obj.get_noise(self.lopt_info) * self.noise_std
            ) # Antithetic sampling
            if epsilon.device != self.meta_params.device:
                self.meta_params.to(epsilon.device)
            mp_plus_epsilon = self.meta_params + epsilon
            mp_minus_epsilon = self.meta_params - epsilon

            if train_x is None or train_y is None or test_x is None or test_y is None:
                train_images, train_labels = next(iter(self.train_loader))
                test_images, test_labels = next(iter(self.ood_val1_loader))
            else:
                train_images, train_labels = train_x[_:(_+1)].squeeze(0), train_y[_:(_+1)].squeeze(0)
                test_images, test_labels = test_x[_:(_+1)].squeeze(0), test_y[_:(_+1)].squeeze(0)

            losses_plus, _ = self.finetune_iter(net, mp_plus_epsilon, self.inner_steps, train_images, train_labels, test_images, test_labels, self.num_iters)
            self.num_iters += self.inner_steps
            losses_minus, _ = self.finetune_iter(net, mp_minus_epsilon, self.inner_steps, train_images, train_labels, test_images, test_labels, self.num_iters)
            self.num_iters += self.inner_steps

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

        net = copy.deepcopy(self.net)
        if train_x is None or train_y is None or test_x is None or test_y is None:
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.ood_val1_loader))
        losses_current, _ = self.finetune_iter(net, self.meta_params, self.inner_steps, train_images, train_labels, test_images,
                                               test_labels, self.num_iters)
        meta_train_loss = losses_current[-1] * self.meta_loss_final_w + losses_current.mean() * self.meta_loss_avg_w
        print(f"Meta-train Loss: {meta_train_loss:.4f}")

        return self.meta_params, grads_mean, meta_train_loss

    def outer_loop_step_parallel(self, net=None, epsilon=None, train_x=None, train_y=None, test_x=None, test_y=None):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
        Parallelizes over tasks in the meta batch using vmap.
        Only pass in params _net, epsilon, train_x, train_y, test_x, test_y when unit-testing."""
        if epsilon is None:
            self.lopt_info["num_features"] = self.meta_batch_size // 2
            epsilon = (
                    self.optimizer_obj.get_noise(self.lopt_info) * self.noise_std
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
        losses_plus, _ = self.finetune(func_net, net_params, mp_plus_epsilon, self.inner_steps, train_images, train_labels, test_images, test_labels)
        losses_minus, _ = self.finetune(func_net, net_params, mp_minus_epsilon, self.inner_steps, train_images, train_labels, test_images, test_labels)
        losses_plus = torch.stack(losses_plus, dim=0).transpose(0, 1)
        losses_minus = torch.stack(losses_minus, dim=0).transpose(0, 1) 
        loss_diff = losses_plus - losses_minus

        objective = (
                loss_diff[:, -1].mean() * self.meta_loss_final_w
                + loss_diff.mean() * self.meta_loss_avg_w
        )
        grad = objective * epsilon / self.noise_std / 2
        grads_mean = grad.mean(dim=0)

        if train_x is None or train_y is None or test_x is None or test_y is None:
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.ood_val1_loader))
        losses_current, _ = self.finetune(func_net, net_params, mp_plus_epsilon, self.inner_steps, train_images, train_labels, test_images, test_labels)
        meta_train_loss = losses_current[-1] * self.meta_loss_final_w + losses_current.mean() * self.meta_loss_avg_w

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        print(f"Meta-train Loss: {meta_train_loss.mean():.4f}")

        return self.meta_params, grads_mean, meta_train_loss.mean()

    def outer_loop_step(self):
        if self.run_parallel:
            return self.outer_loop_step_parallel()
        else:
            return self.outer_loop_step_iter()
    