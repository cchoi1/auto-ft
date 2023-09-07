import torch

def compute_hinge_loss(logits, targets, margin=1.0):
    """
    Compute the multi-class hinge loss between `logits` and the ground truth `targets`.
    Args:
    - logits (torch.Tensor): A tensor of shape (batch_size, C) where C is the number of classes.
    - targets (torch.Tensor): A tensor of shape (batch_size,) containing the ground truth labels.
    - margin (float, optional): SVM margin. Default: 1.0.

    Returns:
    - torch.Tensor: The hinge loss.
    """
    targets = targets.long()
    correct_scores = logits[torch.arange(logits.shape[0]), targets]
    margins = logits - correct_scores[:, None] + margin
    margins[torch.arange(logits.shape[0]), targets] = 0
    loss = torch.relu(margins)
    return loss.sum(dim=1).mean()