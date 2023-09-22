import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_hinge_loss
from clip.loss import ClipLoss

class LearnedLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
        print("Num params", len(self.initial_net_params))
        self.hyperparams = hyperparams.cuda().float()
        self.clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1, use_horovod=False)

    def forward(self, logits, labels, net, unlabeled_logits=None, pseudolabels=None, image_features=None, text_features=None, logit_scale=None):
        ce_loss = F.cross_entropy(logits, labels)
        hinge_loss = compute_hinge_loss(logits, labels)
        entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
        entropy = entropy_all.mean()

        # Memory efficient accumulation
        param_sum = sum(param.numel() for param in net.parameters())
        l1_zero_accum = sum(torch.abs(param).sum() for param in net.parameters()) / param_sum
        l2_zero_accum = sum((param ** 2).sum() for param in net.parameters()) / param_sum
        l1_init_accum = sum(torch.abs(param - init_param).sum() for param, init_param in
                            zip(net.parameters(), self.initial_net_params)) / param_sum
        l2_init_accum = sum(((param - init_param) ** 2).sum() for param, init_param in
                            zip(net.parameters(), self.initial_net_params)) / param_sum

        losses = [ce_loss, hinge_loss, entropy, dcm_loss, l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum]
        if unlabeled_logits is not None and pseudolabels is not None:
            unlabeled_ce_loss = F.cross_entropy(unlabeled_logits, pseudolabels)
            losses.append(unlabeled_ce_loss)
        if image_features is not None and text_features is not None and logit_scale is not None:
            clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
            losses.append(clip_loss)
        loss = torch.dot(torch.stack(losses), self.hyperparams)
        del losses; torch.cuda.empty_cache()

        return loss



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .utils import compute_hinge_loss
#
#
# class LearnedLoss(nn.Module):
#     def __init__(self, hyperparams, initial_net_params):
#         super().__init__()
#         self.initial_net_params = [param.detach().cuda() for param in initial_net_params]
#         print("Num params", len(self.initial_net_params))
#         self.hyperparams = hyperparams.cuda().float()
#
#     def forward(self, outputs, targets, net, use_contrastive_loss=False):
#         if (targets == -1).any():
#             pseudo_labels = torch.argmax(outputs, dim=1)
#             mask = (targets == -1)
#             targets[mask] = pseudo_labels[mask]
#
#         ce_loss = F.cross_entropy(outputs, targets)
#         hinge_loss = compute_hinge_loss(outputs, targets)
#         entropy_all = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1)
#         is_wrong = (outputs.argmax(dim=1) != targets).float().detach()
#         dcm_loss = (is_wrong * entropy_all).mean()
#         entropy = entropy_all.mean()
#         del outputs, entropy_all, is_wrong
#         torch.cuda.empty_cache()
#
#         l1_zero_accum = torch.tensor(0.0).cuda()
#         l2_zero_accum = torch.tensor(0.0).cuda()
#         l1_init_accum = torch.tensor(0.0).cuda()
#         l2_init_accum = torch.tensor(0.0).cuda()
#         for param, init_param in zip(net.parameters(), self.initial_net_params):
#             l1_zero_accum += torch.abs(param).mean()
#             l2_zero_accum += (param ** 2).mean()
#             l1_init_accum += torch.abs(param - init_param).mean()
#             l2_init_accum += ((param - init_param) ** 2).mean()
#
#         losses = torch.stack([
#             ce_loss, hinge_loss, entropy, dcm_loss,
#             l1_zero_accum,
#             l2_zero_accum,
#             l1_init_accum,
#             l2_init_accum
#         ])
#         loss = torch.dot(losses, self.hyperparams)
#
#         return loss