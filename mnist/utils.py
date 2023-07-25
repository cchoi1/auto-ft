from collections import defaultdict
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lopt_info(features, net, args):
    if features is not None:
        num_features = len(features)
        if "pos_enc_cont" in features:
            num_features += 1
        if "pos_enc" in features:
            num_features += 1
    else:
        num_features = len([p for p in net.parameters()])
    lopt_info = {
        "features": features,
        "num_features": num_features,
        "tensor_shapes": [p.data.shape for p in net.parameters()],
        "wnb": args.wnb
    }
    return lopt_info

@torch.no_grad()
def evaluate_net(net, loader):
    """Get test accuracy and losses of net."""
    total, correct_sum, loss_sum = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    net.eval()
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        output = net(x)
        loss = loss_fn(output, labels)
        preds = torch.argmax(output.data, -1)
        total += labels.size(0)
        correct_sum += (preds == labels).sum().item()
        loss_sum += loss.item() * labels.size(0)
    return {"acc": correct_sum / total, "loss": loss_sum / total}

def train(num_epochs, model, meta_params, train_loader, val_loader, test_loader, optimizer_obj, lr, patience, features, l2_lambda=None):
    lopt_info = get_lopt_info(features, model)
    optimizer = optimizer_obj(meta_params, model, lopt_info, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    best_val_acc = 0.0
    train_losses_sum, count = 0.0, 0
    no_improvement = 0
    init_params = [p.clone().detach() for p in model.parameters()]
    total_iters = 0
    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        train_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # L2 regularization towards initial model params
            l2_reg = torch.tensor(0.0, device=device)
            if l2_lambda is not None:
                curr_params = [p.clone().detach() for p in model.parameters()]
                for init_p, curr_p in zip(init_params, curr_params):
                    l2_reg += torch.norm((curr_p - init_p), p=2)  # L2 norm
                loss += l2_lambda * l2_reg

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()
            optimizer.step(curr_loss=loss.item(), iter=total_iters, iter_frac=total_iters / (num_epochs * len(train_loader)))

            train_losses_sum += loss.item() * inputs.size(0)
            count += inputs.size(0)
            total_iters += 1

            val_metrics = evaluate_net(model, val_loader)
            val_loss, val_acc = val_metrics["loss"], val_metrics["acc"]
            test_metrics = evaluate_net(model, test_loader)
            test_loss, test_acc = test_metrics["loss"], test_metrics["acc"]
            train_loss = train_losses_sum / count
            train_losses_sum, count = 0.0, 0
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)
            metrics["test_loss"].append(test_loss)
            metrics["test_acc"].append(test_acc)
            if total_iters % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}. {total_iters} iters. "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping!")
                    return model, metrics

    return model, metrics