import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperopt import fmin, hp, tpe, Trials
from torch.utils.data import DataLoader
from torchvision import datasets

from data.dataloaders import MNISTC, get_transform, get_rotated_mnist, get_colored_mnist
from losses.utils import compute_hinge_loss
from networks import get_pretrained_net_fixed

torch.set_num_threads(1)
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Turn these into options later
batch_size = 128
USE_POINTWISE_WEIGHTS = False 
TRAIN_NOISE_RATIO = 0.0 
NUM_EPOCHS = 10
VAL_FREQ = 10
INNER_LOOP_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = get_pretrained_net_fixed(
    ckpt_path="/iris/u/cchoi1/robust-optimizer/mnist/ckpts",
    dataset_name="svhn",
    output_channels=3,
    train=True,
)
pretrained_net = copy.deepcopy(net)

def get_subset(dataset, num_datapoints):
    set_seed()
    rand_idxs = torch.randperm(len(dataset))[:num_datapoints]
    subset = torch.utils.data.Subset(dataset, rand_idxs)
    return subset

root = "/iris/u/cchoi1/Data"
all_datasets = dict()
source_transform = get_transform("svhn", 3)
source_data = datasets.SVHN(
    root="./data", split="train", download=True, transform=source_transform
)
all_datasets["source"] = get_subset(source_data, 10000)

id_transform = get_transform("mnist", 3)
mnist_train = datasets.MNIST(
    root, train=True, download=True, transform=id_transform
)
mnist_test = datasets.MNIST(
    root, train=False, download=True, transform=id_transform
)
if TRAIN_NOISE_RATIO > 0.0:
    num_noisy = int(len(mnist_train) * TRAIN_NOISE_RATIO)
    noisy_idxs = torch.randperm(len(mnist_train))[:num_noisy]
    targets = mnist_train.targets
    targets[noisy_idxs] = torch.randint(0, 10, (num_noisy,))
    mnist_train.targets = targets

all_datasets["id"] = mnist_train
all_datasets["id_val"] = get_subset(mnist_test, 10000)

ood_transform = get_transform("mnistc", 3)
root_c = os.path.join(root, "MNIST-C")
ood_corruptions = ["brightness", "dotted_line", "fog", "glass_blur", "rotate", "scale", "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag"]
ood_data = MNISTC(
    root_c, corruptions=ood_corruptions, train=True, transform=ood_transform
) 
all_datasets["ood_subset_for_hp"] = get_subset(ood_data, 100) # this is what the meta-learner sees
all_datasets["ood"] = get_subset(ood_data, 10000)

test1_data = MNISTC(
    root_c, corruptions=["motion_blur"], train=False, transform=ood_transform
)
all_datasets["test1"] = get_subset(test1_data, 10000)
test2_data = MNISTC(
    root_c, corruptions=["impulse_noise"], train=False, transform=ood_transform
)
all_datasets["test2"] = get_subset(test2_data, 10000)
test3_data = MNISTC(
    root_c, corruptions=["canny_edges"], train=True, transform=ood_transform
)
all_datasets["test3"] = get_subset(test3_data, 10000)

_, test4_data = get_rotated_mnist(root, id_transform)
all_datasets["test4"] = get_subset(test4_data, 10000)

color_mnist_transform = get_transform("colored_mnist", 3)
_, test5_data = get_colored_mnist(root, color_mnist_transform)
all_datasets["test5"] = get_subset(test5_data, 10000)

class LayerLoss(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams.to(device).float()

    def forward(self, inputs, targets, net, initial_net):
        outputs = net(inputs)
        ce_loss = F.cross_entropy(outputs, targets)
        hinge_loss = compute_hinge_loss(outputs, targets)
        entropy_all = -torch.sum(
            F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1
        )
        is_wrong = (outputs.argmax(dim=1) != targets).float().detach()
        dcm_loss = (is_wrong * entropy_all).mean()
        entropy = entropy_all.mean()

        flat_params = torch.cat([param.flatten() for param in net.parameters()])
        flat_params_init = torch.cat([param.flatten() for param in initial_net.parameters()])
        l1_zero = torch.abs(flat_params).mean()
        l2_zero = torch.pow(flat_params, 2).mean()
        l1_init = torch.abs(flat_params - flat_params_init).mean()
        l2_init = torch.pow(flat_params - flat_params_init, 2).mean()

        stacked_losses = torch.stack([ce_loss, hinge_loss, entropy, dcm_loss, l1_zero, l2_zero, l1_init, l2_init])
        loss = torch.dot(stacked_losses, self.hyperparams.detach())
        return loss


def finetune(net, initial_net, optimizer, loss_fn, dataloader, num_steps=100):
    iters = 0
    while True:
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(x, y, net, initial_net)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            iters += 1
            if iters >= num_steps:
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


def evaluate_hparams(net, hyperparams, datasets, repeats=3, num_steps=300, full_eval=False):
    all_val_results = []
    for _ in range(repeats):
        initial_net = copy.deepcopy(net)
        current_net = copy.deepcopy(net)
        initial_net.to(device)
        current_net.to(device)
        optimizer = torch.optim.SGD(
            current_net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"]
        )
        loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(8)])
        loss_fn = LayerLoss(loss_weight_hparams)
        if "dataw_0" in hyperparams.keys():
            data_weights = torch.tensor([hyperparams[f"dataw_{i}"] for i in range(len(datasets["id"]))])
            sampler = torch.utils.data.WeightedRandomSampler(data_weights, len(datasets["id"]))
            train_loader = DataLoader(
                datasets["id"], batch_size=batch_size, num_workers=0, drop_last=True, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                datasets["id"], batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True
            )
        finetune(current_net, initial_net, optimizer, loss_fn, train_loader, num_steps)

        val_results = dict()
        if full_eval:
            eval_datasets = ["source", "id_val", "ood", "test1", "test2", "test3", "test4", "test5"]
        else:
            eval_datasets = ["ood_subset_for_hp"]
        for name in eval_datasets:
            dataset = datasets[name]
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            loss, accuracy = evaluate(current_net, loader)
            del loader
            val_results[f"{name}_loss"] = loss
            val_results[f"{name}_accuracy"] = accuracy
        all_val_results.append(val_results)
    
    if full_eval:
        for name in eval_datasets:
            losses = [r[f"{name}_loss"] for r in all_val_results]
            accs = [r[f"{name}_accuracy"] for r in all_val_results]
            print(f"{name:10s} loss: {np.mean(losses):.3f} +- {np.std(losses):.3f}  acc: {np.mean(accs):.2f} +- {np.std(accs):.2f}")
        print()
    return all_val_results

    
def hp_objective_fn(hparams):
    _net = copy.deepcopy(net)
    val_results = evaluate_hparams(_net, hparams, all_datasets, repeats=3, num_steps=INNER_LOOP_STEPS)
    val_accs = [r["ood_subset_for_hp_accuracy"] for r in val_results]
    # return -np.min(val_accs)  # maximize accuracy on worst run
    return -np.mean(val_accs)  # maximize average accuracy


# ID + OOD FT baseline. Default hparams and cross-entropy.
id_and_ood_data = torch.utils.data.ConcatDataset([all_datasets["id"], all_datasets["ood_subset_for_hp"]])
all_datasets_w_idood = copy.deepcopy(all_datasets)
all_datasets_w_idood["id"] = id_and_ood_data

default_hparams = {f"lossw_{i}": 0.0 for i in range(8)}
default_hparams["lossw_0"] = 1.0
default_hparams["lr"] = 1e-1
default_hparams["momentum"] = 0.9
print(f"\nID+OOD fine-tune baseline:")
evaluate_hparams(net, default_hparams, all_datasets_w_idood, repeats=3, full_eval=True)

print(f"\nID fine-tune baseline:")
evaluate_hparams(net, default_hparams, all_datasets, repeats=3, full_eval=True)

space = {
    f"lossw_{i}": hp.loguniform(f"lossw_{i}", np.log(1e-4), np.log(10.0))
    for i in range(8)
}
if USE_POINTWISE_WEIGHTS:
    N = len(all_datasets["id"])
    space.update({f"dataw_{i}": hp.uniform(f"dataw_{i}", 0.0, 1.0) for i in range(N)})
space["lr"] = hp.loguniform("lr", np.log(1e-3), np.log(1.0))
space["momentum"] = hp.uniform("momentum", 0.0, 1.0)
trials = Trials()

for max_evals in range(VAL_FREQ, VAL_FREQ * NUM_EPOCHS, VAL_FREQ):
    best_hparams = fmin(
        fn=hp_objective_fn,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    for k, v in best_hparams.items():
        if "dataw" not in k:
            print(f"{k:10s}: {v:.3e}")
    evaluate_hparams(net, best_hparams, all_datasets, repeats=3, full_eval=True)
