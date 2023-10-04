import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_hinge_loss
from clip.loss import ClipLoss

class LearnedLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
        # print("Num params", len(self.initial_net_params))
        self.hyperparams = hyperparams.cuda().float()
        self.param_sum = sum(param.numel() for param in initial_net_params)
        self.clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1, use_horovod=False)

    def forward(self, logits, labels, net, unlabeled_logits=None, pseudolabels=None, image_features=None, text_features=None, logit_scale=None):
        ce_loss = F.cross_entropy(logits, labels)
        hinge_loss = compute_hinge_loss(logits, labels)
        entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
        entropy = entropy_all.mean()

        # Memory efficient accumulation
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

        losses = [ce_loss, hinge_loss, entropy, dcm_loss, l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum]
        if unlabeled_logits is not None and pseudolabels is not None:
            unlabeled_ce_loss = F.cross_entropy(unlabeled_logits, pseudolabels)
            losses.append(unlabeled_ce_loss)
        if image_features is not None and text_features is not None and logit_scale is not None:
            clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
            losses.append(clip_loss)
        loss = torch.dot(torch.stack(losses), self.hyperparams)
        del losses 
        torch.cuda.empty_cache()

        return loss