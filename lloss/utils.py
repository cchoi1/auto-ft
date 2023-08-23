import random

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune(net, initial_net, optimizer, loss_fn, dataloader, max_iters=100):
    iters = 0
    while True:
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(x, y, net, initial_net)

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
        x, y = x.to(device), y.to(device)
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


