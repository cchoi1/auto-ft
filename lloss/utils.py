import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune(net, initial_net, optimizer, loss_fn, dataloader, max_iters=100):
    iters = 0
    while True:
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            loss, _ = loss_fn(x, y, net, initial_net)

            optimizer.zero_grad()
            loss.backward()
            if iters % 1000 == 0:
                print(loss.item())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            iters += 1
            if iters >= max_iters:
                del dataloader
                return net

@torch.no_grad()
def evaluate(net, dataloader):
    losses, predictions, targets = [], [], []
    for x, y in dataloader:
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


def save_hparams(best_hparams, args):
    filename_parts = [
        args.pretrain,
        f"npe{args.num_pretrain_examples}",
        args.id,
        f"nie{args.num_id_examples}",
        f"nive{args.num_id_val_examples}",
        f"niue{args.num_id_unlabeled_examples}",
        f"noe{args.num_ood_examples}",
        f"noue{args.num_ood_unlabeled_examples}",
        f"nofhp{args.num_ood_for_hp}",
        f"nte{args.num_test_examples}",
        f"bs{args.batch_size}",
        f"mi{args.max_iters}",
        f"repeats{args.repeats}"
    ]

    filename = "_".join(filename_parts) + ".pkl"
    filepath = os.path.join(args.save_dir, filename)

    with open(filepath, 'wb') as file:
        pickle.dump(best_hparams, file)

    print(f"Saved best_hparams to {filepath}")


