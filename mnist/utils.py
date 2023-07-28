import os
import random
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(num_epochs, model, meta_params, src_val_loader, train_loader, id_val_loader, ood_val_loader, test_loader, optimizer_obj, val, lr, patience, features, lopt_net_dim, l2_lambda=None, wnb=None):
    lopt_info = get_lopt_info(features, model, lopt_net_dim, wnb)
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

            if total_iters % 100 == 0:
                train_loss = train_losses_sum / count
                train_losses_sum, count = 0.0, 0
                src_val_metrics = evaluate_net(model, src_val_loader)
                src_val_loss, src_val_acc = src_val_metrics["loss"], src_val_metrics["acc"]
                id_val_metrics = evaluate_net(model, id_val_loader)
                id_val_loss, id_val_acc = id_val_metrics["loss"], id_val_metrics["acc"]
                ood_val_metrics = evaluate_net(model, ood_val_loader)
                ood_val_loss, ood_val_acc = ood_val_metrics["loss"], ood_val_metrics["acc"]
                test_metrics = evaluate_net(model, test_loader)
                test_loss, test_acc = test_metrics["loss"], test_metrics["acc"]
                metrics["src_losses"].append(src_val_loss)
                metrics["src_accs"].append(src_val_acc)
                metrics["train_loss"].append(train_loss)
                metrics["id_losses"].append(id_val_loss)
                metrics["id_accs"].append(id_val_acc)
                metrics["ood_losses"].append(ood_val_loss)
                metrics["ood_accs"].append(ood_val_acc)
                metrics["test_losses"].append(test_loss)
                metrics["test_accs"].append(test_acc)
                print(
                    f"Epoch {epoch + 1}/{num_epochs}. {total_iters} iters. "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Src Val Loss: {src_val_loss:.4f} | Src Val Acc: {src_val_acc:.4f} | "
                    f"ID Val Loss: {id_val_loss:.4f} | ID Val Acc: {id_val_acc:.4f} | "
                    f"OOD Val Loss: {ood_val_loss:.4f} | OOD Val Acc: {ood_val_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                val_acc = ood_val_acc if val == "ood" else id_val_acc
                if val_acc > best_val_acc:
                    best_val_acc = ood_val_acc
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping!")
                    return model, metrics

    return model, metrics


def get_per_layer_parameters(model):
    grouped_parameters = OrderedDict()
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]  # Get the layer name from the full parameter name
        if layer_name not in grouped_parameters:
            grouped_parameters[layer_name] = []
        grouped_parameters[layer_name].append(param)
    return grouped_parameters


def get_lopt_info(features, net, lopt_net_dim, use_wnb):
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
        "lopt_net_dim": lopt_net_dim,
        "wnb": use_wnb
    }
    return lopt_info

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_meta_params(opt_trainer, exp_name: str, meta_step: int):
    fn = f"results/{exp_name}/{meta_step}.npy"
    if type(opt_trainer.meta_params) == torch.Tensor:
        meta_params = opt_trainer.meta_params.cpu().detach().numpy()
        np.save(fn, np.array(meta_params))
    else:
        meta_params = [per_param_mp.cpu().detach().numpy() for per_param_mp in opt_trainer.meta_params]
        np.savez(fn, *meta_params)