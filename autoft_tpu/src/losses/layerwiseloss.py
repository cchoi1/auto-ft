import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

from .utils import compute_hinge_loss

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
        self.hyperparams = [hp.float().to(self.device) for hp in hyperparams]

    def forward(self, inputs, targets, net, use_contrastive_loss=False):
        outputs = net(inputs)
        if (targets == -1).any():
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (targets == -1)
            targets[mask] = pseudo_labels[mask]

        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        log_softmax_outputs = F.log_softmax(outputs, dim=1)
        entropy = -(torch.exp(log_softmax_outputs) * log_softmax_outputs).mean(dim=1)
        is_wrong = (outputs.argmax(dim=1) != targets).float()
        dcm_loss = (is_wrong * entropy).mean()
        l1_zero_accum = 0.
        l2_zero_accum = 0.
        l1_init_accum = 0.
        l2_init_accum = 0.
        for param, init_param in zip(net.parameters(), self.initial_net_params):
            l1_zero_accum += torch.abs(param).mean()
            l2_zero_accum += (param ** 2).mean()
            l1_init_accum += torch.abs(param - init_param).mean()
            l2_init_accum += ((param - init_param)**2).mean()

        loss_components = [ce_loss, hinge_loss, entropy, dcm_loss, l1_zero_accum, l2_zero_accum, l1_init_accum,
                           l2_init_accum]
        layerwise_losses = [torch.dot(torch.stack(loss_components), hp) for hp in self.hyperparams]

        return torch.stack(layerwise_losses).mean()
