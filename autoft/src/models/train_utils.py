import random

import numpy as np
import torch
import torch.nn.functional as F
from src.models.modeling import ImageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_layerwise_optimizer(model, hyperparams):
    """
    Create a layer-wise optimizer for the given model.

    :param model: An instance of the ImageClassifier model.
    :param hyperparams: A dictionary of hyperparameters.
    :return: Optimizer object.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = [c for c in model.children()][0]  # Extract from nn.DataParallel
    assert isinstance(model, ImageClassifier), "Expected model to be an instance of ImageClassifier"

    layerwise_params = []
    layer_idx = 0

    # Extract layers from the image_encoder (CLIPEncoder) of the model
    for name, module in model.image_encoder.named_children():
        if name == 'model':
            # Initial Convolutional layer
            params_for_layer = {
                'params': module.visual.conv1.parameters(),
                'lr': hyperparams[f"lr_{layer_idx}"],
                'weight_decay': hyperparams[f"wd_{layer_idx}"]
            }
            layerwise_params.append(params_for_layer)
            layer_idx += 1

            # Layer normalization before the transformer
            params_for_layer = {
                'params': module.visual.ln_pre.parameters(),
                'lr': hyperparams[f"lr_{layer_idx}"],
                'weight_decay': hyperparams[f"wd_{layer_idx}"]
            }
            layerwise_params.append(params_for_layer)
            layer_idx += 1

            # Transformer blocks
            for block in module.visual.transformer.resblocks:
                for layer in block.children():
                    params_for_layer = {
                        'params': layer.parameters(),
                        'lr': hyperparams[f"lr_{layer_idx}"],
                        'weight_decay': hyperparams[f"wd_{layer_idx}"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

    # Classification head of the model
    params_for_layer = {
        'params': model.classification_head.parameters(),
        'lr': hyperparams[f"lr_{layer_idx}"],
        'weight_decay': hyperparams[f"wd_{layer_idx}"]
    }
    layerwise_params.append(params_for_layer)

    optimizer = torch.optim.AdamW(layerwise_params)

    return optimizer


@torch.no_grad()
def evaluate_hp(net, dataloader):
    losses, predictions, targets = [], [], []
    for batch in dataloader:
        if type(batch) == dict:
            x = batch["images"].cuda()
            y = batch["labels"].cuda()
        else:
            x, y = batch
            x, y = x.cuda(), y.cuda()
        outputs = net(x)
        if (y == -1).any(): # if parts of the batch are unlabeled, replace them with their pseudolabels
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (y == -1)
            y[mask] = pseudo_labels[mask]
        loss = F.cross_entropy(outputs, y)

        targets.append(y.cpu())
        losses.append(loss.cpu())
        predictions.append(outputs.argmax(dim=1).cpu())

    losses = torch.stack(losses).mean()
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean() * 100
    del dataloader
    return loss.item(), accuracy.item()

def set_seed(seed=0xc0ffee):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_subset(dataset, num_datapoints):
    rand_idxs = torch.randperm(len(dataset))[:num_datapoints]
    subset = torch.utils.data.Subset(dataset, rand_idxs)
    return subset

def print_hparams(best_hparams):
    print("\nBest Hyperparameters:")
    for key, value in best_hparams.items():
        if not "dataw" in key:
            print(f"{key}: {value:.4f}")
