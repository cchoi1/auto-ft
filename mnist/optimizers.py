""" Definitions for meta-learned optimizers."""
from abc import ABC, abstractstaticmethod

import copy
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedOptimizer(Optimizer, ABC):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    @abstractstaticmethod
    def get_init_meta_params(num_features):
        """
        Abstract static method to be implemented by subclasses.
        Returns the initial meta parameters for the learned optimizer.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def get_noise(num_features):
        """
        Abstract static method to be implemented by subclasses.
        Returns noise to be applied during ES inner loop.
        """
        raise NotImplementedError


class LayerSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per tensor."""

    def __init__(self, meta_params, net, features=None, lr=required):
        # assert meta_params.numel() == 4
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.meta_params = meta_params

    @staticmethod
    def get_init_meta_params(inp_info):
        return torch.zeros(inp_info["num_features"])

    @staticmethod
    def get_noise(inp_info):
        return torch.randn(inp_info["num_features"])

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
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


class PerParamSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per parameter."""

    def __init__(self, meta_params, net, features=None, lr=required):
        # assert meta_params.numel() == 4
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.lr_multiplier = torch.sigmoid(meta_params).to(device)

    @staticmethod
    def get_init_meta_params(inp_info):
        """tensor_shapes: a list of shapes of each tensor in the network."""
        num_params = sum([t.numel() for t in inp_info["tensor_shapes"]])
        return torch.zeros(num_params)

    @staticmethod
    def get_noise(inp_info):
        """tensor_shapes: a list of shapes of each tensor in the network."""
        num_params = sum([t.numel() for t in inp_info["tensor_shapes"]])
        return torch.randn(num_params)

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        start = 0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                end = start + p.numel()
                p_lr_multiplier = self.lr_multiplier[start:end].view(p.shape)
                start = end

                p_lr = group["lr"] * p_lr_multiplier
                p.data.add_(d_p * -p_lr)
        assert end == self.lr_multiplier.numel()
        return loss


class LayerSGDLinear(Optimizer):
    """meta-params: weights of linear layer with depth as input."""

    def __init__(self, meta_params, net, features=None, lr=required):
        # assert meta_params.numel() == 4
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
    def get_init_meta_params(inp_info):
        return torch.zeros(inp_info["num_features"])

    @staticmethod
    def get_noise(inp_info):
        return torch.randn(inp_info["num_features"])

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
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


def get_lopt_net(in_dim, hid_dim=2, out_dim=1):
    return nn.Sequential(
        nn.Linear(in_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim)
    )


class LOptNet(Optimizer):
    """Small 1-layer network that takes in (grad, param, depth) as input."""

    def __init__(self, meta_params, net, features, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list(
            [p for p in net.children() if isinstance(p, nn.Linear)]
        )  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            depth = torch.tensor(depth).float()
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w", "init_param": layer.weight})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b", "init_param": layer.bias})
        super().__init__(param_groups, defaults)

        self.features = features
        self.lopt_net = get_lopt_net(in_dim=len(self.features), hid_dim=64, out_dim=1).to(device)
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        meta_params_split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = meta_params_split_p[i].reshape(p_shapes[i])
        self.loss_ema = 0.0
        self.decay = 0.9

    @staticmethod
    def get_init_meta_params(inp_info):
        dummy_net = get_lopt_net(in_dim=inp_info["num_features"], hid_dim=64, out_dim=1)
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights)

    @staticmethod
    def get_noise(inp_info):
        dummy_net = get_lopt_net(in_dim=inp_info["num_features"], hid_dim=64, out_dim=1)
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes))

    def get_lopt_inputs(self, p, g, p_norm, g_norm, g_norm_avg, depth, wb, dist_init_param, iter, iter_frac, loss, loss_ema, tensor_rank, idx, i, j):
        features = []
        if "p" in self.features:
            features.append(p)
        if "g" in self.features:
            features.append(g)
        if "p_norm" in self.features:
            features.append(p_norm)
        if "g_norm" in self.features:
            features.append(g_norm)
        if "g_norm_avg" in self.features:
            features.append(g_norm_avg)
        if "depth" in self.features:
            features.append(depth)
        if "wb" in self.features:
            features.append(wb)
        if "dist_init_param" in self.features:
            features.append(dist_init_param)
        if "iter" in self.features:
            features.append(iter)
        if "iter_frac" in self.features:
            features.append(iter_frac)
        if "loss" in self.features:
            features.append(loss)
        if "loss_ema" in self.features:
            features.append(loss_ema)
        if "tensor_rank" in self.features:
            features.append(tensor_rank)
        if "idx" in self.features:
            features.append(idx)
        if "i" in self.features:
            features.append(i)
        if "j" in self.features:
            features.append(j)

        return torch.stack(features, dim=1)

    def get_tensor_position(self, index):
        """Returns (layer_num, col) (i.e. (i, j) position) of the tensor at index in lopt_net.
            index: index of the tensor in lopt_net"""
        lopt_param_names = [name for (name, param) in self.lopt_net.named_parameters()]
        tensor_name = lopt_param_names[index]
        layer_num = tensor_name.split(".")[0]
        # find all tensors in lopt_named_params that start with layer_num
        layer_tensors = [name for name in lopt_param_names if name.startswith(layer_num)]
        col = [idx for idx, name in enumerate(layer_tensors) if name == tensor_name][0]
        return float(layer_num), float(col)

    def step(self, curr_loss, iter, iter_frac, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            p = group["params"][0]
            if p.grad is None:
                continue

            p_flat = p.data.flatten()
            g_flat = p.grad.data.flatten()
            p_norm = torch.norm(p.data).flatten().repeat(p_flat.shape[0])
            g_norm = torch.norm(p.grad.data).flatten().repeat(p_flat.shape[0])
            g_norm_avg = torch.norm(p.grad.data, dim=-1).mean(dim=0).repeat(p_flat.shape[0]) # grad norm avg across batch
            depth = group["depth"].repeat(p_flat.shape[0]).to(device)
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            dist_init_param = torch.norm(p.data - group["init_param"]).repeat(p_flat.shape[0])
            loss = torch.tensor(curr_loss).repeat(p_flat.shape[0]).to(device)
            self.loss_ema = self.decay * self.loss_ema + (1 - self.decay) * curr_loss
            loss_ema = torch.tensor(self.loss_ema).repeat(p_flat.shape[0]).to(device)
            iter_num = torch.tensor(iter, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            iter_frac_t = torch.tensor(iter_frac, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            tensor_rank = torch.argmin(torch.tensor(p.shape)).repeat(p_flat.shape[0]).to(device, dtype=p_norm.dtype)
            tensor_idx = torch.tensor(i, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            row, col = self.get_tensor_position(i)
            tensor_row = torch.tensor(row, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            tensor_col = torch.tensor(col, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)

            lopt_inputs = self.get_lopt_inputs(p_flat, g_flat, p_norm, g_norm, g_norm_avg, depth, wb_flat, dist_init_param, iter_num, iter_frac_t, loss, loss_ema, tensor_rank, tensor_idx, tensor_row, tensor_col)
            lopt_inputs = lopt_inputs.cpu()

            with torch.no_grad():
                lopt_outputs = self.lopt_net(lopt_inputs).detach()

            lr_multiplier = torch.sigmoid(lopt_outputs).reshape(p.data.shape).to(device)
            p.data.add_(-p.grad.data * group["lr"] * lr_multiplier)

        return loss


class LargeMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_hidden_layers=3):
        super(LargeMLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class PerParamLOptNet(Optimizer):
    def __init__(self, meta_params, net, features, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list(
            [p for p in net.children() if isinstance(p, nn.Linear)]
        )  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            depth = torch.tensor(depth).float()
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w", "init_param": layer.weight})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b", "init_param": layer.bias})
        super().__init__(param_groups, defaults)

        self.num_params = sum([torch.prod(torch.tensor(p.data.shape)) for p in net.parameters()])
        self.lopt_net = LargeMLP(input_dim=self.num_params, output_dim=self.num_params).to(device)
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        meta_params_split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = meta_params_split_p[i].reshape(p_shapes[i])

    @staticmethod
    def get_init_meta_params(inp_info):
        num_params = sum([torch.prod(torch.tensor(p_shape)) for p_shape in inp_info["tensor_shapes"]])
        dummy_net = LargeMLP(input_dim=num_params, output_dim=num_params).to(device)
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights)

    @staticmethod
    def get_noise(inp_info):
        num_params = sum([torch.prod(torch.tensor(p_shape)) for p_shape in inp_info["tensor_shapes"]])
        dummy_net = LargeMLP(input_dim=num_params, output_dim=num_params).to(device)
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes))

    def step(self, curr_loss, iter, iter_frac, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        all_grads_flat = torch.cat([p.grad.data.flatten() for group in self.param_groups for p in group["params"]]).to(device)
        with torch.no_grad():
            self.lopt_net = self.lopt_net.to(device)
            lopt_outputs = self.lopt_net(all_grads_flat).detach()

        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue
            p_grad_lopt_outputs = lopt_outputs[:torch.prod(torch.tensor(p.grad.data.shape))]
            lr_multiplier = torch.sigmoid(p_grad_lopt_outputs).reshape(p.grad.data.shape).to(device)
            p.data.add_(-p.grad.data * group["lr"] * lr_multiplier)

        return loss