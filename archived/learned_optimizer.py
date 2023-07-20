"""Old code for meta-training several randomly-initialized models in parallel."""

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
from networks import get_pretrained_net, get_pretrained_net_fixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_fn(num_nets: int, ckpt_path: str, train: bool):
    """Combines the states of several pretrained nets together by stacking each parameter.
    Returns a stateless version of the model (func_net) and stacked parameters and buffers."""
    nets = []
    for _ in range(num_nets):
        net = copy.deepcopy(get_pretrained_net(ckpt_path=ckpt_path, train=train))
        nets.append(net)
    func_net, batched_weights, buffers = functorch.combine_state_for_ensemble(nets)
    return func_net, batched_weights, buffers

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

    # def outer_loop_step_iter(self, _net=None, epsilon=None, train_x=None, train_y=None, test_x=None, test_y=None):
    #     """Perform one outer loop step. meta_batch_size tasks with antithetic sampling.
    #     Only pass in params _net, epsilon, train_x, train_y, test_x, test_y when unit-testing."""
    #     grads = []
    #     all_losses_diff = []
    #     for _ in range(self.meta_batch_size // 2):
    #         net = copy.deepcopy(self.net)
    #         if type(self.meta_params) == list:
    #             if epsilon is None:
    #                 epsilon = [noise * self.noise_std for noise in self.optimizer_obj.get_noise(self.lopt_info)]
    #                 # Antithetic sampling
    #             mp_plus_epsilon = [mp + e for mp, e in zip(self.meta_params, epsilon)]
    #             mp_minus_epsilon = [mp - e for mp, e in zip(self.meta_params, epsilon)]
    #             # mp_plus_epsilon = [mp - e for mp, e in zip(self.meta_params, epsilon)]
    #             # mp_minus_epsilon = [mp + e for mp, e in zip(self.meta_params, epsilon)]
    #             mp_plus_avg = torch.tensor([mp.mean() for mp in mp_plus_epsilon]).mean()
    #             mp_minus_avg = torch.tensor([mp.mean() for mp in mp_minus_epsilon]).mean()
    #             # print('mp plus', mp_plus_avg, 'mp minus', mp_minus_avg)
    #         else:
    #             print('hi')
    #             if epsilon is None:
    #                 epsilon = (
    #                         self.optimizer_obj.get_noise(self.lopt_info) * self.noise_std
    #                 ) # Antithetic sampling
    #             mp_plus_epsilon = self.meta_params + epsilon
    #             mp_minus_epsilon = self.meta_params - epsilon
    #             # print('mp_plus', torch.mean(mp_plus_epsilon), 'mp_minus', torch.mean(mp_minus_epsilon))
    #
    #         # breakpoint()
    #         if train_x is None or train_y is None or test_x is None or test_y is None:
    #             train_images, train_labels = next(iter(self.train_loader))
    #             test_images, test_labels = next(iter(self.ood_val1_loader))
    #         else:
    #             train_images, train_labels = train_x[_:(_+1)].squeeze(0), train_y[_:(_+1)].squeeze(0)
    #             test_images, test_labels = test_x[_:(_+1)].squeeze(0), test_y[_:(_+1)].squeeze(0)
    #
    #         losses_plus = self.finetune_iter(net, mp_plus_epsilon, train_images, train_labels, test_images, test_labels, self.num_iters)
    #         self.num_iters += self.inner_steps
    #
    #         losses_minus = self.finetune_iter(net, mp_minus_epsilon, train_images, train_labels, test_images, test_labels, self.num_iters)
    #         self.num_iters += self.inner_steps
    #         print('losses_plus', losses_plus, 'losses_minus', losses_minus)
    #
    #         loss_diff = losses_plus - losses_minus
    #         # breakpoint()
    #         all_losses_diff.append(loss_diff)
    #         objective = (
    #                 loss_diff[-1] * self.meta_loss_final_w
    #                 + loss_diff.mean() * self.meta_loss_avg_w
    #         )
    #         print('obj', objective)
    #         if type(epsilon) == list:
    #             objs = torch.cat([(objective * eps / self.noise_std / 2).flatten() for eps in epsilon])
    #             grads.append(objs)
    #             # for eps in epsilon:
    #             #     grads.append((objective * eps / self.noise_std / 2).flatten())
    #         else:
    #             grads.append(objective * epsilon / self.noise_std / 2)
    #
    #
    #     if type(self.meta_params) == list:
    #         breakpoint()
    #         grads_mean = torch.stack(grads).mean(dim=0)
    #     else:
    #         breakpoint()
    #         grads_mean = torch.stack(grads).mean(dim=0)
    #     print(grads_mean.mean())
    #
    #     self.meta_optimizer.zero_grad()
    #     if type(self.meta_params) == list:
    #         flattened_idxs = [self.meta_params[i].shape.numel() for i in range(len(self.meta_params))]
    #         idx = 0
    #         for i in range(len(self.meta_params)):
    #             self.meta_params[i].grad = grads_mean[idx: idx + flattened_idxs[i]].view(self.meta_params[i].shape)
    #             idx += flattened_idxs[i]
    #     else:
    #         self.meta_params.grad = grads_mean
    #     self.meta_optimizer.step()
    #
    #     return self.meta_params, grads_mean