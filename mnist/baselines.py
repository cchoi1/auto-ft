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
    with torch.no_grad():
        for input, labels in loader:
            input, labels = input.to(device), labels.to(device)
            output = net(input)
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
    losses = []

    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        if i % 100 == 0:
            print(f"Iteration {i} | Loss: {loss.item():.2f}")
        loss.backward()
        losses.append(loss.item())
        inner_opt.step()

    return net, meta_params

def train(num_epochs, model, meta_params, train_loader, val_loader, optimizer_obj, lr, patience, l2_lambda: int =None):
    model = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_obj(meta_params, model, lr=lr)

    val_loss = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    init_params = [p.clone().detach() for p in model.parameters()]
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        train_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # L2-regularization to initial model params
            l2_reg = torch.tensor(0.0, device=device)
            if l2_lambda is not None:
                curr_params = [p.clone().detach() for p in model.parameters()]
                for init_p, curr_p in zip(init_params, curr_params):
                    l2_reg += torch.norm((curr_p - init_p), p=2)  # L2 norm
                loss += l2_lambda * l2_reg

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                print("Early stopping! Validation loss hasn't improved in", patience, "epochs.")
                return model

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    return model


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