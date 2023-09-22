import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from src.losses.utils import compute_hinge_loss


class LearnedLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.device = xm.xla_device()
        self.initial_net_params = [param.clone().detach().to(self.device) for param in initial_net_params]
        self.hyperparams = hyperparams.to(self.device).float()
        self.param_sum = sum(param.numel() for param in initial_net_params)

    def forward(self, outputs, targets, net, use_contrastive_loss=False):
        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy_all = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1)
        is_wrong = (outputs.argmax(dim=1) != targets).float().detach()
        dcm_loss = (is_wrong * entropy_all).mean()
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
            torch.tensor(l1_zero_accum, device=self.device),
            torch.tensor(l2_zero_accum, device=self.device),
            torch.tensor(l1_init_accum, device=self.device),
            torch.tensor(l2_init_accum, device=self.device)
        ])
        loss = torch.dot(losses, self.hyperparams)

        return loss
