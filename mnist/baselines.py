import copy
import numpy as np
import torch
import torch.nn as nn

import optimizers
from mnist import load_dataset
from networks import get_pretrained_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def evaluate_net(net, loader):
    """Get test accuracy and losses of net."""
    accs, losses = [], []
    total, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = net(images)
            loss = loss_fn(output, labels)
            losses.append(loss.item())
            preds = torch.argmax(output.data, -1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total

    return acc, np.array(losses)


def fine_tune_epoch(_net, meta_params, train_loader, optimizer_obj, inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net, train accuracy, and train loss."""
    net = copy.deepcopy(_net)
    inner_opt = optimizer_obj(meta_params, net, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()

    for train_images, train_labels in train_loader:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

    return net, meta_params


def run_baseline(baseline, args):
    print(f"\n--- Baseline: {baseline} ---\n")
    if baseline == "full":
        meta_params = torch.ones(4).float()
    elif baseline == "surgical":
        meta_params = torch.tensor([100, -100, -100, -100]).float()
    else:
        raise ValueError("Baseline must be 'full' or 'surgical'")
    val_losses, test_losses = [], []
    val_accs, test_accs = [], []
    train_loader, val_loader = load_dataset(root_dir=args.data_dir, dataset=args.ft_distribution)
    test_loader, _ = load_dataset(root_dir=args.data_dir, dataset=args.test_distribution)
    for _ in range(args.num_nets):
        _net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        net, meta_params = fine_tune_epoch(
            _net,
            meta_params,
            train_loader,
            optimizers.LayerSGD,
            inner_lr=0.1,
        )
        # Get accuracy and loss on ft distribution.
        acc, losses = evaluate_net(net, val_loader)
        val_accs.append(acc)
        val_losses.append(losses[-1])
        # Get accuracy and loss on test distribution.
        acc, losses = evaluate_net(net, test_loader)
        test_accs.append(acc)
        test_losses.append(losses[-1])

    val_losses, test_losses = np.array(val_losses), np.array(test_losses)
    val_accs, test_accs = np.array(val_accs), np.array(test_accs)
    print(
        f"Val Acc: {val_accs.mean():.4f} +- {val_accs.std():.4f} | Val Loss: {val_losses.mean():.4f} +- {val_losses.std():.4f}")
    print(
        f"Test Acc: {test_accs.mean():.4f} +- {test_accs.std():.4f} | Test Loss: {test_losses.mean():.4f} +- {test_losses.std():.4f}")
