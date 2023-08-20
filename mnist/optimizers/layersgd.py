""" Definitions for meta-learned optimizers that learn per-tensor lr multipliers, updates, etc."""
import torch
from torch.optim.optimizer import Optimizer, required

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per tensor."""

    def __init__(self, meta_params, net, lopt_info=None, lr=required):
        self.lopt_info = lopt_info
        assert meta_params[lopt_info["meta_params"]["start"]:len(meta_params)].numel() == len(list(net.parameters()))
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.initial_weights = [p.data.clone() for p in net.parameters()]
        self.meta_params = meta_params[lopt_info['meta_params']['start']: len(meta_params)]

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

                if self.lopt_info["output"] == "lr_multiplier":
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
                elif self.lopt_info["output"] == "update":
                    o1, o2, _ = self.unpack_meta_params(i)
                    update = -1 * (torch.exp(o1 * 1e-3) * o2 * 1e-3).reshape(p.data.shape).to(device)

                p.data.add_(update)

        return loss