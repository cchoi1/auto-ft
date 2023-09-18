import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_hinge_loss

class LayerwiseLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = [param.clone().detach().cuda() for param in initial_net_params]
        self.hyperparams = [hp.cuda().float() for hp in hyperparams]

    def forward(self, inputs, targets, net, use_contrastive_loss=False):
        outputs = net(inputs)

        if (targets == -1).any():
            pseudo_labels = torch.argmax(outputs, dim=1)
            targets = torch.where(targets == -1, pseudo_labels, targets)

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

        loss_components = [ce_loss, hinge_loss, entropy, dcm_loss, l1_zero_accum, l2_zero_accum, l1_init_accum,
                           l2_init_accum]
        layerwise_losses = [torch.dot(torch.stack(loss_components), hp) for hp in self.hyperparams]

        return torch.stack(layerwise_losses).mean()
