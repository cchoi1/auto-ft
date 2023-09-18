import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_hinge_loss


class LearnedLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
        self.hyperparams = hyperparams.cuda().float()

    def forward(self, inputs, targets, net, use_contrastive_loss=False):
        outputs = net(inputs)

        if (targets == -1).any():
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (targets == -1)
            targets[mask] = pseudo_labels[mask]

        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy_all = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1)
        is_wrong = (outputs.argmax(dim=1) != targets).float().detach()
        dcm_loss = (is_wrong * entropy_all).mean()
        entropy = entropy_all.mean()
        del inputs, outputs, entropy_all, is_wrong
        torch.cuda.empty_cache()

        l1_zero_accum = 0.0
        l2_zero_accum = 0.0
        l1_init_accum = 0.0
        l2_init_accum = 0.0
        for param, init_param in zip(net.parameters(), self.initial_net_params):
            l1_zero_accum += torch.abs(param).mean().item()
            l2_zero_accum += (param ** 2).mean().item()
            l1_init_accum += torch.abs(param - init_param).mean().item()
            l2_init_accum += ((param - init_param) ** 2).mean().item()

        losses = torch.stack([
            ce_loss, hinge_loss, entropy, dcm_loss,
            torch.tensor(l1_zero_accum, device=self.hyperparams.device),
            torch.tensor(l2_zero_accum, device=self.hyperparams.device),
            torch.tensor(l1_init_accum, device=self.hyperparams.device),
            torch.tensor(l2_init_accum, device=self.hyperparams.device)
        ])
        loss = torch.dot(losses, self.hyperparams)

        return loss