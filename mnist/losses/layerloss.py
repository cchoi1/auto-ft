import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import compute_hinge_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerLoss(nn.Module):
    def __init__(self, meta_params, net, lloss_info):
        super(LayerLoss, self).__init__()
        self.loss_weights = {}
        num_tensors = len(list(net.parameters()))
        self.num_loss_weights = 9
        start, end = lloss_info['meta_params']['start'], lloss_info['meta_params']['end']
        self.loss_weights = meta_params[start:end+1].view(num_tensors, self.num_loss_weights).to(device)

    @staticmethod
    def get_init_meta_params(lloss_info):
        num_meta_params = lloss_info['meta_params']['end'] - lloss_info['meta_params']['start'] + 1
        return torch.zeros(num_meta_params)

    @staticmethod
    def get_noise(lloss_info):
        num_meta_params = lloss_info['meta_params']['end'] - lloss_info['meta_params']['start'] + 1
        return torch.randn(num_meta_params)

    def forward(self, outputs, targets, net, pretrained_net=None):
        total_loss = 0
        # 1. Cross-entropy
        ce_loss = F.cross_entropy(outputs, targets)

        # 2. Hinge loss (assuming binary classification here)
        hinge_loss = compute_hinge_loss(outputs, targets)

        # 3. KL towards uniform prediction
        uniform_dist = torch.full_like(outputs, 1.0 / outputs.size(1))
        kl_loss = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dist, reduction='batchmean')

        # 4. Entropy of prediction
        entropy = -torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1).mean()

        # 5. Energy of prediction
        energy = torch.norm(outputs, p=2)

        for i, param in enumerate(net.parameters()):
            # 6. L1 distance to zero
            l1_zero = torch.norm(param.detach(), 1)

            # 7. L2 distance to zero
            l2_zero = torch.norm(param.detach(), 2)

            # 8 & 9. L1 & L2 distance to pretrained params
            if pretrained_net:
                pretrained_param = list(pretrained_net.parameters())[i]
                l1_pretrained = torch.norm(param.detach() - pretrained_param.detach(), 1)
                l2_pretrained = torch.norm(param.detach() - pretrained_param.detach(), 2)
            else:
                l1_pretrained = l2_pretrained = 0

            # Combine losses with learned weights
            combined_loss = torch.stack([ce_loss, hinge_loss, kl_loss, entropy, energy,
                                         l1_zero, l2_zero, l1_pretrained, l2_pretrained])
            total_loss += torch.dot(torch.sigmoid(self.loss_weights[i]), combined_loss)

        return total_loss / (len(list(net.parameters())) * 9)