""" Definitions for meta-learned optimizers that learn per-parameter lr multipliers, updates, etc."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from .utils import compute_positional_encoding, compute_continuous_positional_encoding, get_lopt_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LOptNet2(Optimizer):
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
            param_groups.append(
                {"params": layer.weight, "depth": depth, "type": "w", "init_param": layer.weight.data.clone(),
                 "momentum_buffer": None, "layer_type": layer_type})
            param_groups.append(
                {"params": layer.bias, "depth": depth, "type": "b", "init_param": layer.bias.data.clone(),
                 "momentum_buffer": None, "layer_type": layer_type})
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

        # Calculating the momentum terms
        momentums = [0.5, 0.9, 0.99, 0.999, 0.9999]
        momentum_terms = [g_flat.data.cpu() * m for m in momentums]

        # Calculating the transformed training iteration using tanh squashing function
        timescales = torch.exp(torch.linspace(np.log(3), np.log(300000), 9))
        iterations_transformed = [torch.tanh(iter / eta - 1).unsqueeze(0) for eta in timescales]
        iterations_transformed = torch.stack(iterations_transformed, dim=-1)
        iterations_transformed = iterations_transformed.repeat(p_flat.size(0), 1)

        # Normalizing non-time features by the second moment
        # Assuming that 'batch dimension' refers to the last dimension
        p_normalized_flat = (p.data / (torch.std(p.data, dim=0, keepdim=True) + 1e-8)).view(-1, 1).cpu()
        g_normalized_flat = (g.data / (torch.std(g.data, dim=0, keepdim=True) + 1e-8)).view(-1, 1).cpu()

        # Assembling all features
        features = [p_normalized_flat, g_normalized_flat, iterations_transformed] + momentum_terms

        return torch.cat(features, dim=-1).to(device)

    def unpack_lopt_outputs(self, lopt_outputs, p):
        o1, o2 = lopt_outputs[:, 0], lopt_outputs[:, 1]
        update_coeff = torch.exp(1e-3 * o1) * 1e-3 * o2

        return update_coeff.reshape(p.data.shape).to(device)

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
                lopt_net = self.lopt_net[group['layer_type']][group['type']].to(device)
                lopt_outputs = lopt_net(lopt_inputs).detach()

            update = self.unpack_lopt_outputs(lopt_outputs, p)

            p.data.add_(update)

        return loss
