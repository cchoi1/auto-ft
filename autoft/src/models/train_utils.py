import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def save_hparams(args, best_hparams):
    filename_parts = [args.id, args.ood] + args.eval_datasets + [
        f"nie{args.num_id_examples}",
        f"nive{args.num_id_val_examples}",
        f"noe{args.num_ood_examples}",
        f"nohpe{args.num_ood_hp_examples}",
        f"ep{args.epochs}",
        f"vf{args.val_freq}",
        f"is{args.inner_steps}",
        f"bs{args.batch_size}",
        f"rep{args.repeats}"
    ]

    filename = "_".join(filename_parts) + ".json"
    filepath = os.path.join(args.save, filename)

    with open(filepath, 'w') as file:
        json.dump(best_hparams, file)

    print(f"Saved best_hparams to {filepath}")
