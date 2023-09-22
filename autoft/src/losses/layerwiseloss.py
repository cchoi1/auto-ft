import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_hinge_loss


class LayerwiseLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = [param.clone().detach().cuda() for param in initial_net_params]
        self.hyperparams = torch.stack(hyperparams).cuda()
        self.param_sum = sum(param.numel() for param in initial_net_params)

    def forward(self, logits, labels, net, unlabeled_logits=None, pseudolabels=None):
        ce_loss = F.cross_entropy(logits, labels)
        hinge_loss = compute_hinge_loss(logits, labels)
        entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
        entropy = entropy_all.mean()

        # Memory efficient accumulation
        # param_sum = sum(param.numel() for param in net.parameters())
        # l1_zero_accum = sum(torch.abs(param).sum() for param in net.parameters()) / param_sum
        # l2_zero_accum = sum((param ** 2).sum() for param in net.parameters()) / param_sum
        # l1_init_accum = sum(torch.abs(param - init_param).sum() for param, init_param in
        #                     zip(net.parameters(), self.initial_net_params)) / param_sum
        # l2_init_accum = sum(((param - init_param) ** 2).sum() for param, init_param in
        #                     zip(net.parameters(), self.initial_net_params)) / param_sum
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

        if unlabeled_logits is None and pseudolabels is None:
            losses = torch.stack([
                ce_loss, hinge_loss, entropy, dcm_loss,
                l1_zero_accum, l2_zero_accum,
                l1_init_accum, l2_init_accum
            ])
        else:
            unlabeled_ce_loss = F.cross_entropy(unlabeled_logits, pseudolabels)
            losses = torch.stack([
                ce_loss, unlabeled_ce_loss, hinge_loss, entropy, dcm_loss,
                l1_zero_accum, l2_zero_accum,
                l1_init_accum, l2_init_accum
            ])
        layerwise_losses = torch.matmul(self.hyperparams, losses)

        return layerwise_losses.mean()
