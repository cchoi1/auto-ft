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
        # layers = list(
        #     [p for p in net.children() if isinstance(p, nn.Linear)]
        # )  # Assumes nn.Sequential model
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
        self.lopt_net = get_lopt_net(in_dim=self.lopt_info["input_dim"],
                                     hid_dim=self.lopt_info["hidden_dim"],
                                     out_dim=self.lopt_info["output_dim"]).to(device)

        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]

        # Split meta_params
        split_sizes = p_sizes
        meta_params_splits = meta_params[:sum(split_sizes)].split(split_sizes)

        # Assign to the parameters of the network
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = meta_params_splits[i].reshape(p_shapes[i])

        return

    @staticmethod
    def get_init_meta_params(lopt_info):
        dummy_net = get_lopt_net(
            in_dim=lopt_info["input_dim"],
            hid_dim=lopt_info["hidden_dim"],
            out_dim=lopt_info["output_dim"]
        )
        init_weights = [p.data.flatten() for p in dummy_net.parameters()]

        return torch.cat(init_weights)

    @staticmethod
    def get_noise(lopt_info):
        dummy_net = get_lopt_net(
            in_dim=lopt_info["input_dim"],
            hid_dim=lopt_info["hidden_dim"],
            out_dim=lopt_info["output_dim"]
        )
        p_sizes = [torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()]

        return torch.randn(sum(p_sizes))

    def get_lopt_inputs(self, p, g, group, curr_loss, iter, iter_frac):
        p_flat = p.data.clone().cpu().view(-1, 1)
        g_flat = g.data.clone().cpu().view(-1, 1)

        features = []
        if "p" in self.lopt_info["features"]:
            features.append(p_flat)
        if "g" in self.lopt_info["features"]:
            features.append(g_flat)
        if "p_norm" in self.lopt_info["features"]:
            p_norm = torch.norm(p.data.clone().cpu()) * torch.ones_like(p_flat)
            features.append(p_norm)
        if "g_norm" in self.lopt_info["features"]:
            g_norm = torch.norm(p.grad.data.clone().cpu()) * torch.ones_like(p_flat)
            features.append(g_norm)
        if "g_norm_avg" in self.lopt_info["features"]:
            g_norm_avg = torch.norm(p.grad.data.clone().cpu(), dim=-1).mean(dim=0) * torch.ones_like(p_flat)  # grad norm avg across batch
            features.append(g_norm_avg)
        if "depth" in self.lopt_info["features"]:
            depth = group["depth"] * torch.ones_like(p_flat)
            features.append(depth)
        if "wb" in self.lopt_info["features"]:
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb * torch.ones_like(p_flat), dtype=p_flat.dtype)
            features.append(wb_flat)
        if "layer_type" in self.lopt_info["features"]:
            layer_type = 0 if group["layer_type"] == "linear" else 1
            layer_type_flat = torch.tensor(layer_type * torch.ones_like(p_flat), dtype=p_flat.dtype)
            features.append(layer_type_flat)
        if "dist_init_param" in self.lopt_info["features"]:
            dist_init_param = torch.norm(p.data.clone().cpu() - group["init_param"].cpu(), keepdim=True) * torch.ones_like(p_flat)
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
            loss = torch.tensor(curr_loss.clone()) * torch.ones_like(p_flat)
            features.append(loss)
        if "loss_ema" in self.lopt_info["features"]:
            loss_ema = torch.tensor(self.loss_ema.clone()) * torch.ones_like(p_flat)
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
            betas = [0.5, 0.9, 0.99, 0.999, 0.9999]
            if group["momentum_buffer"] is None:
                group["momentum_buffer"] = {}
                for beta in betas:
                    group["momentum_buffer"][beta] = torch.zeros_like(p.data)
            for beta in betas:
                group["momentum_buffer"][beta].mul_(beta).add_((1 - beta) * g.clone().data)
                m_flat = group["momentum_buffer"][beta].clone().cpu().view(-1, 1)
                features.append(m_flat)

        return torch.cat(features, dim=-1)

    def unpack_lopt_outputs(self, lopt_outputs, p):
        o1 = lopt_outputs[:, 0].reshape(p.data.shape).to(device)
        o2, momentum_multiplier = None, None
        if self.lopt_info["output_dim"] == 3:
            o2 = lopt_outputs[:, 1].reshape(p.data.shape).to(device)
            momentum_multiplier = lopt_outputs[:, 2].reshape(p.data.shape).to(device)
        elif self.lopt_info["output_dim"] == 2:
            if self.lopt_info["wnb"]:
                o2 = lopt_outputs[:, 1].reshape(p.data.shape).to(device)
            elif self.lopt_info["momentum"]:
                momentum_multiplier = lopt_outputs[:, 1].reshape(p.data.shape).to(device)

        return o1, o2, momentum_multiplier

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
                lopt_net = self.lopt_net.to(device)
                lopt_outputs = lopt_net(lopt_inputs).detach()

            o1, o2, mom_multiplier = self.unpack_lopt_outputs(lopt_outputs, p)
            if self.lopt_info["output"] == "lr_multiplier":
                if o2 is not None:
                    local_lr = torch.sigmoid(o1 * group["lr"] + o2)
                else:
                    local_lr = torch.sigmoid(o1) * group["lr"]
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
                update = (torch.exp(o1 * 1e-3) * o2 * 1e-3).reshape(p.data.shape).to(device)

            p.data.add_(update)

        return loss
