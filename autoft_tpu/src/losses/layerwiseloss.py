import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

from .utils import compute_hinge_loss


class LayerwiseLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.device = xm.xla_device()
        self.initial_net_params = [param.clone().detach().to(self.device) for param in initial_net_params]
        self.hyperparams = torch.stack(hyperparams).to(self.device)
        self.param_sum = sum(param.numel() for param in initial_net_params)

    def forward(self, outputs, targets, net, use_contrastive_loss=False):
        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy_all = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1)
        dcm_loss = ((outputs.argmax(dim=1) != targets).float().detach() * entropy_all).mean()
        entropy = entropy_all.mean()

        l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum = 0, 0, 0, 0
        for param, init_param in zip(net.parameters(), self.initial_net_params):
            l1_zero_accum += torch.abs(param).sum()
            l2_zero_accum += (param ** 2).sum()

            diff = param - init_param
            l1_init_accum += torch.abs(diff).sum()
            l2_init_accum += (diff ** 2).sum()

        l1_zero_accum /= self.param_sum
        l2_zero_accum /= self.param_sum
        l1_init_accum /= self.param_sum
        l2_init_accum /= self.param_sum

        losses = torch.stack([
            ce_loss, hinge_loss, entropy, dcm_loss,
            l1_zero_accum,
            l2_zero_accum,
            l1_init_accum,
            l2_init_accum
        ])
        layerwise_losses = torch.matmul(self.hyperparams, losses)

        return torch.mean(layerwise_losses)
