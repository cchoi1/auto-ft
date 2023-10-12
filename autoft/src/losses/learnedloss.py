import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_hinge_loss
from clip.loss import ClipLoss
from src.models import utils

class LearnedLoss(nn.Module):
    def __init__(self, loss_terms, hyperparams, initial_net_params):
        super().__init__()
        self.loss_terms = loss_terms
        self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
        if isinstance(hyperparams, list):
            hyperparams = torch.stack(hyperparams)
        self.hyperparams = hyperparams.cuda().float()
        self.param_sum = sum(param.numel() for param in initial_net_params)
        self.clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1, use_horovod=False)

    def forward(self, model, logits, labels, image_features, text_features=None, logit_scale=None, unlabeled_image_features=None, pseudolabels=None):
        losses = []
        entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        if "ce" in self.loss_terms:
            ce_loss = F.cross_entropy(logits, labels)
            losses.append(ce_loss)
        if "hinge" in self.loss_terms:
            hinge_loss = compute_hinge_loss(logits, labels)
            losses.append(hinge_loss)
        if "entropy" in self.loss_terms:
            entropy = entropy_all.mean()
            losses.append(entropy)
        if "dcm" in self.loss_terms:
            dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
            losses.append(dcm_loss)

        l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum = 0, 0, 0, 0
        for param, init_param in zip(model.parameters(), self.initial_net_params):
            l1_zero_accum += torch.abs(param).sum()
            l2_zero_accum += (param ** 2).sum()
            diff = param - init_param
            l1_init_accum += torch.abs(diff).sum()
            l2_init_accum += (diff ** 2).sum()
        l1_zero_accum /= self.param_sum
        l2_zero_accum /= self.param_sum
        l1_init_accum /= self.param_sum
        l2_init_accum /= self.param_sum
        if "l1zero" in self.loss_terms:
            losses.append(l1_zero_accum)
        if "l2zero" in self.loss_terms:
            losses.append(l2_zero_accum)
        if "l1init" in self.loss_terms:
            losses.append(l1_init_accum)
        if "l2init" in self.loss_terms:
            losses.append(l2_init_accum)
        del l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum; torch.cuda.empty_cache()

        if unlabeled_image_features is not None and pseudolabels is not None:
            unlabeled_logits = model.classification_head(unlabeled_image_features)
            unlabeled_ce_loss = F.cross_entropy(unlabeled_logits, pseudolabels)
            losses.append(unlabeled_ce_loss)
        if "flyp" in self.loss_terms:
            clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
            losses.append(clip_loss)
        losses = torch.matmul(self.hyperparams, torch.stack(losses))

        return losses.mean()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .utils import compute_hinge_loss
# from clip.loss import ClipLoss
#
# class LearnedLoss(nn.Module):
#     def __init__(self, loss_terms, hyperparams, initial_net_params):
#         super().__init__()
#         self.loss_terms = loss_terms
#         self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
#         if isinstance(hyperparams, list):
#             hyperparams = torch.stack(hyperparams)
#         self.hyperparams = hyperparams.cuda().float()
#         self.param_sum = sum(param.numel() for param in initial_net_params)
#         self.clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1, use_horovod=False)
#
#     def forward(self, logits, labels, net, unlabeled_logits=None, pseudolabels=None, image_features=None, text_features=None, logit_scale=None):
#         losses = []
#         entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
#         if "ce" in self.loss_terms:
#             ce_loss = F.cross_entropy(logits, labels)
#             losses.append(ce_loss)
#         if "hinge" in self.loss_terms:
#             hinge_loss = compute_hinge_loss(logits, labels)
#             losses.append(hinge_loss)
#         if "entropy" in self.loss_terms:
#             entropy = entropy_all.mean()
#             losses.append(entropy)
#         if "dcm" in self.loss_terms:
#             dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
#             losses.append(dcm_loss)
#
#         l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum = 0, 0, 0, 0
#         for param, init_param in zip(net.parameters(), self.initial_net_params):
#             l1_zero_accum += torch.abs(param).sum()
#             l2_zero_accum += (param ** 2).sum()
#             diff = param - init_param
#             l1_init_accum += torch.abs(diff).sum()
#             l2_init_accum += (diff ** 2).sum()
#         l1_zero_accum /= self.param_sum
#         l2_zero_accum /= self.param_sum
#         l1_init_accum /= self.param_sum
#         l2_init_accum /= self.param_sum
#         if "l1_zero" in self.loss_terms:
#             losses.append(l1_zero_accum)
#         if "l2_zero" in self.loss_terms:
#             losses.append(l2_zero_accum)
#         if "l1_init" in self.loss_terms:
#             losses.append(l1_init_accum)
#         if "l2_init" in self.loss_terms:
#             losses.append(l2_init_accum)
#
#         if unlabeled_logits is not None and pseudolabels is not None:
#             unlabeled_ce_loss = F.cross_entropy(unlabeled_logits, pseudolabels)
#             losses.append(unlabeled_ce_loss)
#         if image_features is not None and text_features is not None and logit_scale is not None:
#             clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
#             losses.append(clip_loss)
#         losses = torch.matmul(self.hyperparams, torch.stack(losses))
#         del losses; torch.cuda.empty_cache()
#
#         return losses.mean()