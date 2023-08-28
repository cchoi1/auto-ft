import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .utils import compute_hinge_loss

devices = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")

class LayerLoss(nn.Module):
    def __init__(self, hyperparams, initial_net):
        super().__init__()
        self.initial_net = initial_net
        self.hyperparams = hyperparams.cuda().float()

    def forward(self, inputs, targets, net, use_contrastive_loss=False):
        outputs = net(inputs)
        if targets is None:
            targets = torch.argmax(outputs, dim=1)

        losses = []
        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy = -torch.sum(
            F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1
        ).mean()
        del outputs

        flat_params = torch.cat([param.flatten() for param in net.parameters()])
        flat_params_init = torch.cat([param.flatten() for param in self.initial_net.parameters()])
        l1_zero = torch.abs(flat_params).mean()
        l2_zero = torch.pow(flat_params, 2).mean()
        l1_init = torch.abs(flat_params - flat_params_init).mean()
        l2_init = torch.pow(flat_params - flat_params_init, 2).mean()

        losses.extend([ce_loss, hinge_loss, entropy, l1_zero, l2_zero, l1_init, l2_init])

        if use_contrastive_loss:
            image_features = net.encode_image(inputs["image"])
            text_features = net.encode_text(inputs["text"])
            logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
            logits_per_text = torch.matmul(text_features, image_features.t()) / self.temperature
            contrastive_loss = (F.cross_entropy(logits_per_image, torch.arange(len(logits_per_image), device=device)) +
                                F.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=device))) / 2
            losses.append(contrastive_loss)

        stacked_losses = torch.stack(losses)
        loss = torch.dot(stacked_losses, self.hyperparams.detach())
        del losses

        return loss, stacked_losses.detach()