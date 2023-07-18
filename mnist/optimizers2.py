from abc import ABC, abstractstaticmethod

import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from optimizers import get_lopt_net
from loss_state import _clip_log_abs, _fractional_tanh_embed, _sorted_values, BufferLossAccumulators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LOptNet2(Optimizer):

    def __init__(self, meta_params, net, features, lr=required, num_steps=required):
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

        self.buffer_loss_fns = BufferLossAccumulators()
        self.loss_state = self.buffer_loss_fns.init(num_steps)
        self.loss_features = self.buffer_loss_fns.features(self.buffer_loss_fns.init(10))

        self.features = features
        self.lopt_net = get_lopt_net(len(self.features)).to(device)
        p_shapes = [p.data.shape for p in self.lopt_net.parameters()]
        p_sizes = [torch.prod(torch.tensor(s)) for s in p_shapes]
        split_p = meta_params.split(p_sizes)
        for i, p in enumerate(self.lopt_net.parameters()):
            p.data = split_p[i].reshape(p_shapes[i])

    @staticmethod
    def get_init_meta_params(num_features):
        dummy_net = get_lopt_net(num_features)
        dummy_params = [p for p in dummy_net.parameters()]
        init_weights = [p.data.flatten() for p in dummy_params]
        return torch.cat(init_weights)

    @staticmethod
    def get_noise(num_features):
        dummy_net = get_lopt_net(num_features)
        p_sizes = [
            torch.prod(torch.tensor(p.data.shape)) for p in dummy_net.parameters()
        ]
        return torch.randn(sum(p_sizes))

    def get_lopt_inputs(self, p, g, m, rms, depth, wb, dist_init_param, fraction_trained, loss_features):
        norm_mult = torch.rsqrt(torch.maximum(1e-9, torch.mean(p ** 2)))
        g = g * norm_mult
        p = p * norm_mult
        m = m * norm_mult
        rms = rms * norm_mult

        inputs = {}

        fraction_left = _fractional_tanh_embed(fraction_trained)
        inputs["fraction_left"] = fraction_left
        inputs["loss_features"] = loss_features

        leading_axis = list(range(0, len(p.shape)))
        mean_m = torch.mean(m, dim=leading_axis, keepdim=True)
        var_m = torch.mean(torch.square(m - mean_m), dim=leading_axis)
        inputs["var_m"] = _clip_log_abs(var_m, scale=10.)

        mean_rms = torch.mean(rms, dim=leading_axis, keepdim=True)
        var_rms = torch.mean(torch.square(rms - mean_m), dim=leading_axis)
        inputs["mean_rms"] = _clip_log_abs(torch.reshape(mean_rms, [mean_rms.shape[-1]]), scale=10.)
        inputs["var_rms"] = _clip_log_abs(var_rms, scale=10.)

        # tensor rank
        n_rank = torch.sum(torch.tensor(p.shape) > 1).item()
        inputs["rank"] = torch.nn.functional.one_hot(torch.tensor(n_rank), 5)

        values = _sorted_values(inputs)
        values = [v if len(v.shape) == 1 else torch.unsqueeze(v, 0) for v in values]

        return torch.cat(values, dim=0).to(device)

    def step(self, curr_loss_state, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        next_loss_buffer = self.buffer_loss_fns.update(
            curr_loss_state.loss_buffer, loss)
        loss_features = self.buffer_loss_fns.features(next_loss_buffer)

        for group in self.param_groups:
            p = group["params"][0]
            mom = group["momentum"]

            if p.grad is None:
                continue

            p_flat = p.data.flatten()
            g_flat = p.grad.data.flatten()
            depth = group["depth"].repeat(p_flat.shape[0]).to(device)
            wb = 0 if group["type"] == "w" else 1
            wb_flat = torch.tensor(wb, dtype=p_flat.dtype).repeat(p_flat.shape[0]).to(device)
            dist_init_param = torch.norm(p_flat - group["init_param"].flatten()).repeat(p_flat.shape[0]).to(device)

            breakpoint()
            lopt_inputs = self.get_lopt_inputs(p_flat, g_flat, depth, wb_flat, dist_init_param, loss_features)
            self.lopt_net = self.lopt_net.to(device)
            with torch.no_grad():
                lopt_outputs = self.lopt_net(lopt_inputs).detach()

            lr_multiplier = torch.sigmoid(lopt_outputs).reshape(p.data.shape)
            p.data = p.data - p.grad.data * group["lr"] * lr_multiplier
            d_p = p.grad.data
            param_state = self.state[p]

            # Initialize momentum buffer if not already present
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            else:
                buf = param_state['momentum_buffer']
            # Update the momentum buffer
            buf.mul_(mom).add_(d_p)

        return loss