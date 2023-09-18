import copy
from collections import defaultdict

import torch
import torch.nn as nn

from lloss.losses.layerloss import LayerLoss
from utils import get_lopt_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_epochs, model, meta_params, src_val_loader, train_loader, id_val_loader, ood_val_loader, test_loader,
          loss_fn, optimizer_obj, lr, val, alpha, args):
    # Instantiation of optimizer
    lopt_info = get_lopt_info(model, args) if optimizer_obj else None
    optimizer = optimizer_obj(meta_params, model, lopt_info, lr=lr) if optimizer_obj else torch.optim.SGD(
        model.parameters(), lr=lr)

    metrics = defaultdict(list)
    train_losses_sum, count = 0.0, 0
    best_val_acc, no_improvement = 0.0, 0
    pretrained_net = copy.deepcopy(model)
    init_params = [p.clone().detach() for p in model.parameters()]
    total_iters = 0
    param_avg = {name: p.clone().detach() for name, p in model.named_parameters()} if args.method == "wise-ft" else None

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Instantiation of loss
            loss = loss_fn(outputs, labels, model, pretrained_net) if isinstance(loss_fn, LayerLoss) else loss_fn(
                outputs, labels)
            if args.l2_lambda:
                loss += compute_l2_regularization(init_params, model, args.l2_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer_step_args = {'curr_loss': loss.item(), 'iter': total_iters,
                                   'iter_frac': total_iters / (num_epochs * len(train_loader))} if optimizer_obj else {}
            optimizer.step(**optimizer_step_args)
            if args.method == "wise-ft":
                update_wise_ft_parameters(model, param_avg, alpha)

            train_losses_sum += loss.item() * len(inputs)
            count += len(inputs)
            if total_iters % args.ft_val_freq == 0:
                train_loss = train_losses_sum / count
                train_losses_sum, count = 0.0, 0
                val_metrics, original_params = get_val_metrics(args.method, model, param_avg, alpha, {
                    'src': src_val_loader,
                    'id': id_val_loader,
                    'ood': ood_val_loader,
                    'test': test_loader
                })

                for key, metric in val_metrics.items():
                    metrics[f"{key}_losses"].append(metric["loss"])
                    metrics[f"{key}_accs"].append(metric["acc"])

                val_acc = val_metrics[val]['acc']
                if total_iters % 100 == 0:
                    best_val_acc, no_improvement = (val_acc, 0) if val_acc > best_val_acc else (
                    best_val_acc, no_improvement + 1)
                    if no_improvement >= args.patience:
                        print(f"Early stopping!")
                        return model, metrics

                if args.method == "wise-ft":
                    restore_original_parameters(model, original_params)

                print_metrics(val_metrics, epoch, num_epochs, total_iters, train_loss)

            total_iters += 1

    return model, metrics

def compute_l2_regularization(init_params, model, l2_lambda):
    l2_reg = torch.tensor(l2_lambda, device=device)
    curr_params = [p.clone().detach() for p in model.parameters()]
    for init_p, curr_p in zip(init_params, curr_params):
        l2_reg += torch.norm((curr_p - init_p), p=2)  # L2 norm
    return l2_lambda * l2_reg

def restore_original_parameters(model, original_params):
    for orig_param, curr_param in zip(original_params, model.parameters()):
        curr_param.data.copy_(orig_param)

def update_wise_ft_parameters(model, param_avg, alpha):
    for name, param in model.named_parameters():
        param_avg[name] = alpha * param_avg[name] + (1.0 - alpha) * param.detach()

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

def get_val_metrics(method, model, param_avg, alpha, loaders):
    if method == "wise-ft":
        original_params = [p.clone().detach() for p in model.parameters()]
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(alpha * param + (1.0 - alpha) * param_avg[name])

    metrics = {}
    for key, loader in loaders.items():
        metrics[key] = evaluate_net(model, loader)
    model.train()
    return metrics, original_params if method == "wise-ft" else None

def print_metrics(val_metrics, epoch, num_epochs, total_iters, train_loss):
    print(f"Epoch {epoch + 1}/{num_epochs}. {total_iters} iters. Train Loss: {train_loss:.4f} | ", end="")
    for key, metric in val_metrics.items():
        print(f"{key.capitalize()} Loss: {metric['loss']:.4f} | {key.capitalize()} Acc: {100*metric['acc']:.2f} | ", end="")
    print()