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

def compute_l2_regularization(init_params, model, l2_lambda):
    l2_reg = torch.tensor(l2_lambda, device=device)
    curr_params = [p.clone().detach() for p in model.parameters()]
    for init_p, curr_p in zip(init_params, curr_params):
        l2_reg += torch.norm((curr_p - init_p), p=2)  # L2 norm
    return l2_lambda * l2_reg


def update_wise_ft_parameters(model, param_avg, alpha):
    for name, param in model.named_parameters():
        param_avg[name] = alpha * param_avg[name] + (1.0 - alpha) * param.detach()


def get_val_metrics(method, model, param_avg, alpha, loaders):
    if method == "wise-ft":
        original_params = [p.clone().detach() for p in model.parameters()]
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(alpha * param + (1.0 - alpha) * param_avg[name])

    metrics = {}
    for key, loader in loaders.items():
        metrics[key] = evaluate_net(model, loader)
    return metrics, original_params if method == "wise-ft" else None


def restore_original_parameters(model, original_params):
    for orig_param, curr_param in zip(original_params, model.parameters()):
        curr_param.data.copy_(orig_param)


def train(num_epochs, model, meta_params, src_val_loader, train_loader, id_val_loader, ood_val_loader, test_loader, optimizer_obj, lr, val, alpha, args):
    lopt_info = get_lopt_info(model, args)
    optimizer = optimizer_obj(meta_params, model, lopt_info, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    best_val_acc = 0.0
    train_losses_sum, count = 0.0, 0
    no_improvement = 0
    init_params = [p.clone().detach() for p in model.parameters()]
    total_iters = 0

    param_avg = None
    if args.method == "wise-ft":
        param_avg = {name: param.clone().detach() for name, param in model.named_parameters()}

    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            if args.l2_lambda is not None:
                loss += compute_l2_regularization(init_params, model, args.l2_lambda)

            optimizer.zero_grad()  # Clear gradients
            loss.backward()
            optimizer.step(curr_loss=loss.item(), iter=total_iters, iter_frac=total_iters / (num_epochs * len(train_loader)))

            if args.method == "wise-ft":
                update_wise_ft_parameters(model, param_avg, alpha)

            train_losses_sum += loss.item() * inputs.size(0)
            count += inputs.size(0)

            if total_iters % args.ft_val_freq == 0:
                train_loss = train_losses_sum / count
                train_losses_sum, count = 0.0, 0

                val_metrics, original_params = get_val_metrics(args.method, model, param_avg, alpha, {
                    'src_val': src_val_loader,
                    'id_val': id_val_loader,
                    'ood_val': ood_val_loader,
                    'test': test_loader
                })

                for key, metric in val_metrics.items():
                    metrics[f"{key}_losses"].append(metric["loss"])
                    metrics[f"{key}_accs"].append(metric["acc"])

                print(
                    f"Epoch {epoch + 1}/{num_epochs}. {total_iters} iters. "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Src Val Loss: {val_metrics['src_val']['loss']:.4f} | Src Val Acc: {val_metrics['src_val']['acc']:.4f} | "
                    f"ID Val Loss: {val_metrics['id_val']['loss']:.4f} | ID Val Acc: {val_metrics['id_val']['acc']:.4f} | "
                    f"OOD Val Loss: {val_metrics['ood_val']['loss']:.4f} | OOD Val Acc: {val_metrics['ood_val']['acc']:.4f} | "
                    f"Test Loss: {val_metrics['test']['loss']:.4f} | Test Acc: {val_metrics['test']['acc']:.4f}")

                if total_iters % 100 == 0:
                    # Early stopping based on validation accuracy
                    val_acc = val_metrics['ood_val']['acc'] if val == "ood" else val_metrics['id_val']['acc']
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    if no_improvement >= args.patience:
                        print(f"Early stopping!")
                        return model, metrics

                if args.method == "wise-ft":
                    restore_original_parameters(model, original_params)

            total_iters += 1

    return model, metrics

def get_per_layer_parameters(model):
    grouped_parameters = OrderedDict()
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]  # Get the layer name from the full parameter name
        if layer_name not in grouped_parameters:
            grouped_parameters[layer_name] = []
        grouped_parameters[layer_name].append(param)
    return grouped_parameters


def get_lopt_info(net, args):
    if args.features is not None:
        input_dim = len(args.features)
        if "pos_enc_cont" in args.features:
            input_dim += 1
        if "pos_enc" in args.features:
            input_dim += 1
        if "iter" in args.features:
            input_dim += 8
        if "momentum" in args.features:
            input_dim += 4
    else:
        input_dim = len([p for p in net.parameters()])

    if args.output == "update":
        output_dim = 2
    else:
        output_dim = 1
        if args.wnb:
            output_dim += 1
        if args.momentum:
            output_dim += 1

    lopt_info = {
        "features": args.features,
        "input_dim": input_dim,
        "hidden_dim": args.lopt_net_dim,
        "output_dim": output_dim,
        "tensor_shapes": [p.data.shape for p in net.parameters()],
        "wnb": args.wnb,
        "momentum": args.momentum,
        "output": args.output,
    }
    return lopt_info

def get_lloss_info(net, args):
    lloss_info = {}
    return lloss_info


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