import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datasets import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 64)  # Input size: 28x28=784, Output size: 64
        self.fc2 = nn.Linear(64, 10)  # Input size: 64, Output size: 10

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten input images
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)  # Input size: 28x28=784, Output size: 64

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten input images
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def get_network(dataset_name: str):
    if dataset_name == "svhn":
        net = CNN(num_classes=10)
    elif dataset_name == "mnist":
        net = MLP()
    else:
        raise ValueError(f"Unsupported pretraining dataset {dataset_name}.")
    return net.to(device)


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


def pretrain_net(dataset_name, data_dir, seed=0, lr=1e-3, num_epochs=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = get_network(dataset_name)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

    print(f"Pretraining on {dataset_name}...")
    train_loader, test_loader = get_dataloaders(root_dir=data_dir, dataset_names=[dataset_name], batch_size=64,
                                                meta_batch_size=None, num_workers=0)

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
        scheduler.step()

    accuracy = get_test_accuracy(test_loader=test_loader, model=net)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

    return net


def pretrain_nets(ckpt_path, dataset_name, data_dir, num_nets, num_epochs):
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(exist_ok=True)
    for seed in range(num_nets):
        filename = ckpt_path / f"pretrain_{dataset_name}_{seed}.pt"
        if not filename.exists():
            net = pretrain_net(dataset_name, data_dir, seed=seed, num_epochs=num_epochs)
            torch.save(net.state_dict(), filename)
            print(f"Saved pretrained net to {filename}!")


def get_pretrained_net_fixed(ckpt_path, dataset_name, train):
    """Return a fixed pretrained net."""
    ckpt_path = Path(ckpt_path)
    all_ckpts = glob(str(ckpt_path / f"pretrain_{dataset_name}*.pt"))
    n_ckpts = len(all_ckpts)
    train_N = int(n_ckpts * 0.8)
    train_ckpts, test_ckpts = all_ckpts[:train_N], all_ckpts[train_N:]
    if train:
        checkpoint = torch.load(train_ckpts[0])
    else:
        checkpoint = torch.load(test_ckpts[0])
    net = get_network(dataset_name)
    net.load_state_dict(checkpoint)
    net = net.to(device)
    return net
