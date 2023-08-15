""" Definitions for meta-learned optimizers that learn per-parameter lr multipliers, updates, etc."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from .utils import compute_positional_encoding, compute_continuous_positional_encoding, get_lopt_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LOptNet(Optimizer):
    """Small 1-layer network that takes in (grad, param, depth) as input."""

    def __init__(self, meta_params, net, lopt_info, lr=required):
        defaults = dict(lr=lr)
        param_groups = []
        layers = list(
            [p for p in net.children() if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d)]
        )  # Assumes nn.Sequential model
        for depth, layer in enumerate(layers):
            depth = torch.tensor(depth).float()
            layer_type = 'conv' if isinstance(layer, nn.Conv2d) else 'linear'
            param_groups.append({"params": layer.weight, "depth": depth, "type": "w", "init_param": layer.weight.data.clone(), "momentum_buffer": None, "layer_type": layer_type})
            param_groups.append({"params": layer.bias, "depth": depth, "type": "b", "init_param": layer.bias.data.clone(), "momentum_buffer": None, "layer_type": layer_type})
        super().__init__(param_groups, defaults)

        self.loss_ema = 0.0
        self.decay = 0.9
        self.lopt_info = lopt_info
        self.init_lopt_net(meta_params)

    def init_lopt_net(self, meta_params):
        self.lopt_net = defaultdict(dict)
        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                self.lopt_net[layer_type][param_type] = get_lopt_net(in_dim=self.lopt_info["input_dim"],
                                                                     hid_dim=self.lopt_info["hidden_dim"],
                                                                     out_dim=self.lopt_info["output_dim"]).to(device)
        p_shapes = defaultdict(dict)
        p_sizes = defaultdict(dict)
        meta_params_splits = defaultdict(dict)

        # Calculate shapes and sizes
        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                p_shapes[layer_type][param_type] = [p.data.shape for p in
                                                    self.lopt_net[layer_type][param_type].parameters()]
                p_sizes[layer_type][param_type] = [torch.prod(torch.tensor(s)) for s in
                                                   p_shapes[layer_type][param_type]]

        # Split meta_params accordingly
        offset = 0
        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                split_sizes = p_sizes[layer_type][param_type]
                meta_params_splits[layer_type][param_type] = meta_params[offset:offset + sum(split_sizes)].split(
                    split_sizes)
                offset += sum(split_sizes)

        # Assign to the parameters of the respective networks
        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                for i, p in enumerate(self.lopt_net[layer_type][param_type].parameters()):
                    p.data = meta_params_splits[layer_type][param_type][i].reshape(p_shapes[layer_type][param_type][i])

        return

    @staticmethod
    def get_init_meta_params(lopt_info):
        dummy_nets = defaultdict(dict)
        init_weights = []

        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                dummy_nets[layer_type][param_type] = get_lopt_net(
                    in_dim=lopt_info["input_dim"],
                    hid_dim=lopt_info["hidden_dim"],
                    out_dim=lopt_info["output_dim"]
                )
                init_weights += [p.data.flatten() for p in dummy_nets[layer_type][param_type].parameters()]

        return torch.cat(init_weights)

    @staticmethod
    def get_noise(lopt_info):
        dummy_nets = defaultdict(dict)
        p_sizes = []

        for layer_type in ["linear", "conv"]:
            for param_type in ["w", "b"]:
                dummy_nets[layer_type][param_type] = get_lopt_net(
                    in_dim=lopt_info["input_dim"],
                    hid_dim=lopt_info["hidden_dim"],
                    out_dim=lopt_info["output_dim"]
                )
                p_sizes += [
                    torch.prod(torch.tensor(p.data.shape)) for p in dummy_nets[layer_type][param_type].parameters()
                ]

        return torch.randn(sum(p_sizes))

    def get_lopt_inputs(self, p, g, group, curr_loss, iter, iter_frac):
        p_flat = p.data.view(-1, 1).cpu()
        g_flat = g.data.view(-1, 1).cpu()

        features = []
        if "p" in self.lopt_info["features"]:
            p_normalized_flat = (p.data / (torch.std(p.data, dim=0, keepdim=True) + 1e-8)).view(-1, 1).cpu()
            features.append(p_normalized_flat)
            # features.append(p_flat)
        if "g" in self.lopt_info["features"]:
            g_normalized_flat = (g.data / (torch.std(g.data, dim=0, keepdim=True) + 1e-8)).view(-1, 1).cpu()
            features.append(g_normalized_flat)
            # features.append(g_flat)
        if "p_norm" in self.lopt_info["features"]:
            p_norm = torch.norm(p.data.cpu()) * torch.ones_like(p_flat)
            features.append(p_norm)
        if "g_norm" in self.lopt_info["features"]:
            g_norm = torch.norm(p.grad.data.cpu()) * torch.ones_like(p_flat)
            features.append(g_norm)
        if "g_norm_avg" in self.lopt_info["features"]:
            g_norm_avg = torch.norm(p.grad.data.cpu(), dim=-1).mean(dim=0) * torch.ones_like(p_flat)  # grad norm avg across batch
            features.append(g_norm_avg)
        if "depth" in self.lopt_info["features"]:
            depth = group["depth"] * torch.ones_like(p_flat)
            features.append(depth)
        if "wb" in self.lopt_info["features"]:
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb * torch.ones_like(p_flat), dtype=p_flat.dtype)
            features.append(wb_flat)
        if "dist_init_param" in self.lopt_info["features"]:
            dist_init_param = torch.norm(p.data.cpu() - group["init_param"].cpu()) * torch.ones_like(p_flat)
            features.append(dist_init_param)
        if "iter" in self.lopt_info["features"]:
            timescales = torch.exp(torch.linspace(np.log(3), np.log(300000), 9))
            iterations_transformed = [torch.tanh(iter / eta - 1).unsqueeze(0) for eta in timescales]
            iterations_transformed = torch.stack(iterations_transformed, dim=-1)
            iterations_transformed = iterations_transformed.repeat(p_flat.size(0), 1)
            features.append(iterations_transformed)
            # iter_num = torch.tensor(iter, dtype=p_flat.dtype) * torch.ones_like(p_flat)
            # features.append(iter_num)
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
        if "momentum" in self.lopt_info["features"]:
            momentums = [0.5, 0.9, 0.99, 0.999, 0.9999]
            for m in momentums:
                features.append(m * g_flat.data.cpu())

        return torch.cat(features, dim=-1).to(device)

    def unpack_lopt_outputs(self, lopt_outputs, p):
        lr_multiplier = lopt_outputs[:, 0].reshape(p.data.shape).to(device)
        lr_bias, mom_multiplier = None, None
        if self.lopt_info["output_dim"] == 3:
            lr_bias = lopt_outputs[:, 1].reshape(p.data.shape).to(device)
            mom_multiplier = lopt_outputs[:, 2].reshape(p.data.shape).to(device)
        elif self.lopt_info["output_dim"] == 2:
            if self.lopt_info["wnb"]:
                lr_bias = lopt_outputs[:, 1].reshape(p.data.shape).to(device)
            elif self.lopt_info["momentum"]:
                mom_multiplier = lopt_outputs[:, 1].reshape(p.data.shape).to(device)

        return lr_multiplier, lr_bias, mom_multiplier

    def step(self, curr_loss, iter, iter_frac, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            p = group["params"][0]
            if p.grad is None:
                continue
            d_p = p.grad.data

            self.loss_ema = self.decay * self.loss_ema + (1 - self.decay) * curr_loss
            lopt_inputs = self.get_lopt_inputs(p, p.grad, group, curr_loss, iter, iter_frac).to(device)

            with torch.no_grad():
                # Get the correct optimizer network based on layer_type and param_type from the group
                lopt_net = self.lopt_net[group['layer_type']][group['type']].to(device)
                lopt_outputs = lopt_net(lopt_inputs).detach()

            if self.lopt_info["output"] == "lr_multiplier":
                lr_multiplier, lr_bias, mom_multiplier = self.unpack_lopt_outputs(lopt_outputs, p)
                local_lr = torch.sigmoid(group["lr"] * lr_multiplier)

                if lr_bias is not None:
                    local_lr = torch.sigmoid(lr_bias + group["lr"] * lr_multiplier)
                if mom_multiplier is not None:
                    mom_multiplier = torch.sigmoid(mom_multiplier)
                    if group["momentum_buffer"] is None:
                        group["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        group["momentum_buffer"].mul_(mom_multiplier).add_((1 - mom_multiplier) * d_p)
                    momentum_updated_d_p = group["momentum_buffer"]
                else:
                    momentum_updated_d_p = d_p
                update = -local_lr * momentum_updated_d_p
            elif self.lopt_info["output"] == "update":
                o1, o2 = lopt_outputs[:, 0], lopt_outputs[:, 1]
                update = -1 * (torch.exp(o1 * 1e-3) * o2 * 1e-3).reshape(p.data.shape).to(device)

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