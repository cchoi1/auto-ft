#%%
from typing import Any, Dict

import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required


class LayerSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per parameter."""

    def __init__(self, meta_params, params, lr=required):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.meta_params = meta_params

    @staticmethod
    def get_init_meta_params():
        return torch.zeros(4)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        return super().__setstate__(state)

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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        return super().__setstate__(state)

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
