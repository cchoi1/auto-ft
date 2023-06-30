from abc import ABC, abstractstaticmethod

import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required


class LearnedOptimizer(Optimizer, ABC):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    @abstractstaticmethod
    def get_init_meta_params():
        """
        Abstract static method to be implemented by subclasses.
        Returns the initial meta parameters for the learned optimizer.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def get_noise():
        """
        Abstract static method to be implemented by subclasses.
        Returns noise to be applied during ES inner loop.
        """
        raise NotImplementedError


class LayerSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per parameter."""

    def __init__(self, meta_params, net, lr=required):
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.meta_params = meta_params

    @staticmethod
    def get_init_meta_params():
        return torch.zeros(4)

    @staticmethod
    def get_noise():
        return torch.randn(4)

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
    """meta-params: weights of linear layer with depth as input."""

    def __init__(self, meta_params, net, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list(
            [p for p in net.children() if isinstance(p, nn.Linear)]
        )  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w"})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b"})
        super().__init__(param_groups, defaults)
        self.meta_params = {"w": meta_params[0:2], "b": meta_params[2:4]}

    @staticmethod
    def get_init_meta_params():
        return torch.zeros(4)

    @staticmethod
    def get_noise():
        return torch.randn(4)

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


def get_lopt_net():
    in_dim = 2
    hid = 2
    return nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 1))


class LOptNet(Optimizer):
    """Small 1-layer network that takes in (grad, param, depth) as input."""

    def __init__(self, meta_params, net, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list(
            [p for p in net.children() if isinstance(p, nn.Linear)]
        )  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            depth = torch.tensor(depth).float()
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w"})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b"})
        super().__init__(param_groups, defaults)

        self.lopt_net = get_lopt_net()
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = split_p[i].reshape(p_shapes[i])

    @staticmethod
    def get_init_meta_params():
        dummy_net = get_lopt_net()
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights)

    @staticmethod
    def get_noise():
        dummy_net = get_lopt_net()
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue

            p_flat = p.data.flatten()
            g_flat = p.grad.data.flatten()
            depth = group["depth"].repeat(p_flat.shape[0])
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb).repeat(p_flat.shape[0])
            # lopt_inputs = torch.stack([g_flat, p_flat, depth, wb_flat], dim=1)
            # lopt_inputs = torch.stack([g_flat, depth, wb_flat], dim=1)
            lopt_inputs = torch.stack([depth, wb_flat], dim=1)
            with torch.no_grad():
                lopt_outputs = self.lopt_net(lopt_inputs).detach()

            # update = torch.tanh(lopt_outputs).reshape(p.data.shape)
            # p.data = p.data + group["lr"] * update

            lr_multiplier = torch.sigmoid(lopt_outputs).reshape(p.data.shape)
            p.data = p.data - p.grad.data * group["lr"] * lr_multiplier
        return loss