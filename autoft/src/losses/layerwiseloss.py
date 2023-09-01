import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_hinge_loss

devices = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")

class LayerwiseLoss(nn.Module):
    def __init__(self, hyperparams, initial_net_params):
        super().__init__()
        self.initial_net_params = initial_net_params
        self.hyperparams = [hp.cuda().float() for hp in hyperparams]

    def forward(self, inputs, targets, net, use_contrastive_loss=False):
        outputs = net(inputs)
        if targets is None:
            targets = torch.argmax(outputs, dim=1)

        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy_all = -torch.sum(
            F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1
        )
        is_wrong = (outputs.argmax(dim=1) != targets).float().detach()
        dcm_loss = (is_wrong * entropy_all).mean()
        entropy = entropy_all.mean()
        del outputs

        with torch.no_grad():
            flat_params = torch.cat([param.flatten() for param in net.parameters()])
            flat_params_init = torch.cat([param.flatten() for param in self.initial_net_params])
        l1_zero = torch.abs(flat_params).mean()
        l2_zero = torch.pow(flat_params, 2).mean()
        l1_init = torch.abs(flat_params - flat_params_init).mean()
        l2_init = torch.pow(flat_params - flat_params_init, 2).mean()

        # List to hold final losses for each layer
        layerwise_losses = []

        for layer_hyperparams in self.hyperparams:
            losses = [ce_loss, hinge_loss, entropy, dcm_loss, l1_zero, l2_zero, l1_init, l2_init]

            if use_contrastive_loss:
                image_features = net.encode_image(inputs["image"])
                text_features = net.encode_text(inputs["text"])
                logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
                logits_per_text = torch.matmul(text_features, image_features.t()) / self.temperature
                contrastive_loss = (F.cross_entropy(logits_per_image, torch.arange(len(logits_per_image), device=device)) +
                                    F.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=device))) / 2
                losses.append(contrastive_loss)

            stacked_losses = torch.stack(losses)
            loss = torch.dot(stacked_losses, layer_hyperparams.detach())
            layerwise_losses.append(loss)

        return torch.stack(layerwise_losses), torch.stack(layerwise_losses)

