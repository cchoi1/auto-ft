import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision import datasets, transforms
from src.models.utils import initialize_model
from argparse import Namespace

DATA_DIR = "/home/carolinechoi/robust-ft"
def get_data_loaders(batch_size, preprocess_fn):
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=preprocess_fn)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=4)

    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=preprocess_fn)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=4)

    return train_loader, test_loader


def _run(index, flags, args):
    device = xm.xla_device()
    model, preprocess_fn = initialize_model(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.AdamW(model.parameters(), lr=flags['lr'], weight_decay=flags['wd'])

    train_loader, test_loader = get_data_loaders(flags['batch_size'], preprocess_fn)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    test_loader = pl.MpDeviceLoader(test_loader, device)

    for epoch in range(flags['epochs']):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)

            if i % 100 == 0:
                xm.master_print(f"Epoch [{epoch + 1}/{flags['epochs']}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")

        xm.master_print("Finished training epoch:", epoch + 1)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    xm.master_print(f"Accuracy on test data: {accuracy * 100}%")


# Main function
def main():
    flags = {
        'batch_size': 16,
        'lr': 3e-5,
        'wd': 0.1,
        'epochs': 1,
    }
    args = Namespace()
    args.load = "/home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt"
    args.freeze_encoder = False
    xmp.spawn(_run, args=(flags, args), nprocs=8, start_method='spawn')  # Assuming an 8-core TPU


if __name__ == "__main__":
    main()
