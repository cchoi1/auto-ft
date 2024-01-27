import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_hinge_loss
from clip.loss import ClipLoss


class LearnedLoss(nn.Module):
    def __init__(self, losses, loss_weights, initial_params):
        super().__init__()
        self.losses = losses
        self.initial_params = [param.detach().cuda() for param in initial_params]
        if isinstance(loss_weights, list):
            loss_weights = torch.stack(loss_weights)
        self.loss_weights = loss_weights.cuda().float()
        self.param_sum = sum(param.numel() for param in initial_params)
        self.device = initial_params[0].device
        if "flyp" in self.losses:
            self.clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0,
                                        world_size=1, use_horovod=False)

    def forward(self, model, logits, labels, image_features, text_features=None, logit_scale=None, unlabeled_logits=None, pseudolabels=None):
        losses = []
        if "ce" in self.losses:
            ce_loss = F.cross_entropy(logits, labels)
            if unlabeled_logits is not None and pseudolabels is not None:
                ce_loss += F.cross_entropy(unlabeled_logits, pseudolabels)
            losses.append(ce_loss)
        if "dcm" in self.losses or "entropy" in self.losses:
            entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        if "dcm" in self.losses:
            dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
            losses.append(dcm_loss)
        if "entropy" in self.losses:
            entropy = entropy_all.mean()
            losses.append(entropy)
        if "flyp" in self.losses:
            clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
            losses.append(clip_loss)
        if "hinge" in self.losses:
            hinge_loss = compute_hinge_loss(logits, labels)
            losses.append(hinge_loss)

        if "l1zero" in self.losses or "l2zero" in self.losses or "l1init" in self.losses or "l2init" in self.losses:
            l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum = 0, 0, 0, 0
            for param, init_param in zip(model.parameters(), self.initial_params):
                l1_zero_accum += torch.abs(param).sum()
                l2_zero_accum += (param ** 2).sum()
                diff = param - init_param
                l1_init_accum += torch.abs(diff).sum()
                l2_init_accum += (diff ** 2).sum()
            l1_zero_accum /= self.param_sum
            l2_zero_accum /= self.param_sum
            l1_init_accum /= self.param_sum
            l2_init_accum /= self.param_sum
            if "l1init" in self.losses:
                losses.append(l1_init_accum)
            if "l1zero" in self.losses:
                losses.append(l1_zero_accum)
            if "l2init" in self.losses:
                losses.append(l2_init_accum)
            if "l2zero" in self.losses:
                losses.append(l2_zero_accum)
            del l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum
            torch.cuda.empty_cache()

        losses = torch.matmul(self.loss_weights, torch.stack(losses))

        return losses.mean()