""" Definitions for meta-learned optimizers that learn per-tensor lr multipliers, updates, etc."""
import torch
from torch.optim.optimizer import Optimizer, required

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per tensor."""

    def __init__(self, meta_params, net, lopt_info=None, lr=required):
        self.lopt_info = lopt_info
        assert meta_params.numel() == lopt_info["output_dim"] * lopt_info["input_dim"]
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.meta_params = meta_params
        self.initial_weights = [p.data.clone() for p in net.parameters()]

    @staticmethod
    def get_init_meta_params(lopt_info):
        return torch.zeros(lopt_info["output_dim"] * lopt_info["input_dim"])

    @staticmethod
    def get_noise(lopt_info):
        return torch.randn(lopt_info["output_dim"] * lopt_info["input_dim"])

    def unpack_meta_params(self, layer):
        lr_multiplier = torch.sigmoid(self.meta_params[self.lopt_info["output_dim"] * layer])
        lr_bias, mom_multiplier = None, None
        if self.lopt_info["output_dim"] == 2:
            if self.lopt_info["wnb"]:
                lr_bias = torch.sigmoid(self.meta_params[self.lopt_info["output_dim"] * layer + 1])
            elif self.lopt_info["momentum"]:
                mom_multiplier = torch.sigmoid(self.meta_params[self.lopt_info["output_dim"] * layer + 1])
        elif self.lopt_info["output_dim"] == 3:
            lr_bias = torch.sigmoid(self.meta_params[self.lopt_info["output_dim"] * layer + 1])
            mom_multiplier = torch.sigmoid(self.meta_params[self.lopt_info["output_dim"] * layer + 2])

        return lr_multiplier, lr_bias, mom_multiplier

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                lr_multiplier, lr_bias, mom_multiplier = self.unpack_meta_params(i)
                local_lr = group["lr"] * lr_multiplier
                if lr_bias is not None:
                    local_lr += lr_bias
                if mom_multiplier is not None:
                    state = self.state.get(i)
                    if state is None:
                        state = {}
                        self.state[i] = state
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state['momentum_buffer'] = buf
                    else:
                        buf.mul_(mom_multiplier).add_((1 - mom_multiplier) * d_p)
                    d_p = buf
                update = -local_lr * d_p

                p.data.add_(update)

        return loss

# class LayerSGDLinear(Optimizer):
#     """meta-params: weights of linear layer with depth as input."""
#
#     def __init__(self, meta_params, net, lopt_info=None, lr=required):
#         # assert meta_params.numel() == 4
#         defaults = dict(lr=lr)
#         param_groups = []
#         layers = list(
#             [p for p in net.children() if isinstance(p, nn.Linear)]
#         )  # Assumes nn.Sequential model
#         for depth, layer in enumerate(layers):
#             param_groups.append({"params": layer.weight, "depth": depth, "type": "w"})
#             param_groups.append({"params": layer.bias, "depth": depth, "type": "b"})
#         super().__init__(param_groups, defaults)
#         self.meta_params = {"w": meta_params[0:2], "b": meta_params[2:4]}
#
#     @staticmethod
#     def get_init_meta_params(lopt_info):
#         return torch.zeros(lopt_info["input_dim"])
#
#     @staticmethod
#     def get_noise(lopt_info):
#         return torch.randn(lopt_info["input_dim"])
#
#     def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             p = group["params"][0]
#             if p.grad is None:
#                 continue
#
#             depth = group["depth"]
#             meta_params = self.meta_params[group["type"]]
#             lr_multiplier = torch.sigmoid(meta_params[0] * depth + meta_params[1])
#             p.data.add_(p.grad.data, alpha=-group["lr"] * lr_multiplier)
#         return loss
#
#