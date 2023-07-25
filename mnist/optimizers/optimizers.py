""" Definitions for meta-learned optimizers."""
from abc import ABC, abstractstaticmethod

import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from .utils import compute_positional_encoding, compute_continuous_positional_encoding, get_lopt_net

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

    def __init__(self, meta_params, net, lopt_info=None, lr=required):
        self.lopt_info = lopt_info
        if lopt_info["wnb"]:
            assert meta_params.numel() == 2 * lopt_info["num_features"]
        else:
            assert meta_params.numel() == lopt_info["num_features"]
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.meta_params = meta_params

    @staticmethod
    def get_init_meta_params(lopt_info):
        if lopt_info["wnb"]:
            return torch.zeros(2 * lopt_info["num_features"])
        return torch.zeros(lopt_info["num_features"])

    @staticmethod
    def get_noise(lopt_info):
        if lopt_info["wnb"]:
            return torch.randn(2 * lopt_info["num_features"])
        return torch.randn(lopt_info["num_features"])

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if self.lopt_info["wnb"]:
                    # grad clipping?? b/c inner loop loss immediately shoots up to NaN on SVHN task
                    # torch.nn.utils.clip_grad_norm_(p, 1.0)
                    local_lr = group["lr"] * torch.sigmoid(self.meta_params[2*i]) + torch.sigmoid(self.meta_params[2*i+1])
                else:
                    local_lr = group["lr"] * torch.sigmoid(self.meta_params[i])
                p.data.add_(d_p, alpha=-local_lr)
        return loss

class LayerSGDLinear(Optimizer):
    """meta-params: weights of linear layer with depth as input."""

    def __init__(self, meta_params, net, lopt_info=None, lr=required):
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
    def get_init_meta_params(lopt_info):
        return torch.zeros(lopt_info["num_features"])

    @staticmethod
    def get_noise(lopt_info):
        return torch.randn(lopt_info["num_features"])

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


class LOptNet(Optimizer):
    """Small 1-layer network that takes in (grad, param, depth) as input."""

    def __init__(self, meta_params, net, lopt_info, lr=required):
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

        self.lopt_info = lopt_info
        out_dim = 2 if self.lopt_info["wnb"] else 1
        self.lopt_net = get_lopt_net(in_dim=self.lopt_info["num_features"], hid_dim=2, out_dim=out_dim).to(device)
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        meta_params_split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = meta_params_split_p[i].reshape(p_shapes[i])
        self.loss_ema = 0.0
        self.decay = 0.9

    @staticmethod
    def get_init_meta_params(lopt_info):
        out_dim = 2 if lopt_info["wnb"] else 1
        dummy_net = get_lopt_net(in_dim=lopt_info["num_features"], hid_dim=2, out_dim=out_dim)
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights)

    @staticmethod
    def get_noise(lopt_info):
        out_dim = 2 if lopt_info["wnb"] else 1
        dummy_net = get_lopt_net(in_dim=lopt_info["num_features"], hid_dim=2, out_dim=out_dim)
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes))

    def get_lopt_inputs(self, p, g, group, curr_loss, iter, iter_frac):
        p_flat = p.data.view(-1, 1).cpu()
        g_flat = g.data.view(-1, 1).cpu()

        features = []
        if "p" in self.lopt_info["features"]:
            features.append(p_flat)
        if "g" in self.lopt_info["features"]:
            features.append(g_flat)
        if "p_norm" in self.lopt_info["features"]:
            p_norm = torch.norm(p.data) * torch.ones_like(p_flat)
            features.append(p_norm)
        if "g_norm" in self.lopt_info["features"]:
            g_norm = torch.norm(p.grad.data) * torch.ones_like(p_flat)
            features.append(g_norm)
        if "g_norm_avg" in self.lopt_info["features"]:
            g_norm_avg = torch.norm(p.grad.data, dim=-1).mean(dim=0)  # grad norm avg across batch
            features.append(g_norm_avg)
        if "depth" in self.lopt_info["features"]:
            depth = group["depth"] * torch.ones_like(p_flat)
            features.append(depth)
        if "wb" in self.lopt_info["features"]:
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb * torch.ones_like(p_flat), dtype=p_flat.dtype)
            features.append(wb_flat)
        if "dist_init_param" in self.lopt_info["features"]:
            dist_init_param = torch.norm(p.data - group["init_param"]) * torch.ones_like(p_flat)
            features.append(dist_init_param)
        if "iter" in self.lopt_info["features"]:
            iter_num = torch.tensor(iter, dtype=p_flat.dtype) * torch.ones_like(p_flat)
            features.append(iter_num)
        if "iter_frac" in self.lopt_info["features"]:
            iter_frac_t = torch.tensor(iter_frac, dtype=p_flat.dtype) * torch.ones_like(p_flat)
            features.append(iter_frac_t)
        if "loss" in self.lopt_info["features"]:
            loss = torch.tensor(curr_loss) * torch.ones_like(p_flat)
            features.append(loss)
        if "loss_ema" in self.lopt_info["features"]:
            loss_ema = torch.tensor(self.loss_ema) * torch.ones_like(p_flat)
            features.append(loss_ema)
        if "tensor_rank" in self.lopt_info["features"]:
            tensor_rank = torch.argmin(torch.tensor(p.shape)) * torch.ones_like(p_flat)
            features.append(tensor_rank)
        if "pos_enc" in self.lopt_info["features"]:
            param_pos = compute_positional_encoding(p.shape).float()
            features.append(param_pos)
        if "pos_enc_cont" in self.lopt_info["features"]:
            param_pos_cont = compute_continuous_positional_encoding(p.shape, d_model=2).float()
            features.append(param_pos_cont)

        return torch.cat(features, dim=-1).to(device)

    def step(self, curr_loss, iter, iter_frac, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            p = group["params"][0]
            if p.grad is None:
                continue

            self.loss_ema = self.decay * self.loss_ema + (1 - self.decay) * curr_loss

            lopt_inputs = self.get_lopt_inputs(p, p.grad, group, curr_loss, iter, iter_frac).to(device)
            with torch.no_grad():
                self.lopt_net = self.lopt_net.to(device)
                lopt_outputs = self.lopt_net(lopt_inputs).detach()
            if self.lopt_info["wnb"]:
                w = torch.sigmoid(lopt_outputs[:, 0]).reshape(p.data.shape).to(device)
                b = torch.sigmoid(lopt_outputs[:, 1]).reshape(p.data.shape).to(device)
                update = -p.grad.data * group["lr"] * w + b
            else:
                lr_multiplier = torch.sigmoid(lopt_outputs).reshape(p.data.shape).to(device)
                update = -p.grad.data * lr_multiplier
            p.data.add_(update)

        return loss


class GlobalLOptNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GlobalLOptNetMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass the input (gradients) through the MLP
        g = torch.relu(self.fc1(x))
        g = torch.sigmoid(self.fc2(g))

        # Compute the residual channel y = x + g * x
        y = x + (g * x)
        return y


class GlobalLOptNet(Optimizer):
    def __init__(self, meta_params, net, lopt_info, lr=required):
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)

        self.num_params = sum([torch.prod(torch.tensor(p.data.shape)) for p in net.parameters()])
        self.lopt_net = GlobalLOptNetMLP(input_dim=self.num_params, hidden_dim=2, output_dim=self.num_params).to(device)
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        meta_params_split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = meta_params_split_p[i].reshape(p_shapes[i])

    @staticmethod
    def get_init_meta_params(lopt_info):
        num_params = sum([torch.prod(torch.tensor(p_shape)) for p_shape in lopt_info["tensor_shapes"]])
        dummy_net = GlobalLOptNetMLP(input_dim=num_params, hidden_dim=2, output_dim=num_params)
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights).to(device)

    @staticmethod
    def get_noise(lopt_info):
        num_params = sum([torch.prod(torch.tensor(p_shape)) for p_shape in lopt_info["tensor_shapes"]])
        dummy_net = GlobalLOptNetMLP(input_dim=num_params, hidden_dim=2, output_dim=num_params)
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes)).to(device)

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