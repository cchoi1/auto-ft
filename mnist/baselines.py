import copy
import numpy as np
import torch
import torch.nn as nn

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


def full_fine_tune(_net, train_loader, lr=1e-1):
    """Fine-tune net on train_loader."""
    net = copy.deepcopy(_net)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for train_images, train_labels in train_loader:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return net

def surgical_fine_tune(_net, train_loader, lr=1e-1):
    """Fine-tune the first layer of net on train_loader."""
    net = copy.deepcopy(_net)
    # Freeze all layers except the first layer
    for param in net.parameters():
        param.requires_grad = False

    # Set the first layer to be trainable
    net.fc1.weight.requires_grad = True
    net.fc1.bias.requires_grad = True

    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for train_images, train_labels in train_loader:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return net