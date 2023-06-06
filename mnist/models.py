import argparse
import copy
import random
import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from mnist import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_network(hidden=50):
    net = ConvNet().to(device)
    return net

def get_test_accuracy(test_loader, model):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy

def pretrain_net(seed=0, lr=1e-3, num_epochs=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = get_network()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    train_loader, test_loader = load_mnist(dataset="mnist", batch_size=64)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            opt.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    accuracy = get_test_accuracy(test_loader=test_loader, model=net)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

    return net

def pretrain_nets(ckpt_path, num_nets):
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(exist_ok=True)
    for seed in range(num_nets):
        filename = ckpt_path / f"pretrain_{seed}.pt"
        if not filename.exists():
            net = pretrain_net(seed=seed)
            torch.save(net.state_dict(), filename)
            print(f"Saved pretrained net to {filename}!")

def get_pretrained_net(ckpt_path, train):
    """Return a randomly sampled pretrained net."""
    ckpt_path = Path(ckpt_path)
    all_ckpts = glob(str(ckpt_path / "pretrain_*.pt"))
    n_ckpts = len(all_ckpts)
    train_N = int(n_ckpts * 0.8)
    train_ckpts, test_ckpts = all_ckpts[:train_N], all_ckpts[train_N:]
    if train:
        random_fn = random.choice(train_ckpts)
    else:
        random_fn = random.choice(test_ckpts)
    rand_checkpoint = torch.load(random_fn)
    net = get_network()
    net.load_state_dict(rand_checkpoint)
    net = net.to(device)
    return net