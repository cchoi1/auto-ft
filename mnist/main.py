import copy
import importlib
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import wandb

import optimizers
from data.dataloaders import get_dataloaders
from learned_optimizer import OptimizerTrainer
from networks import get_pretrained_net_fixed, pretrain_nets
from parser import get_args
from utils import evaluate_net, train, save_meta_params, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_optimizer(args):
    """ Train optimizer and return meta-params. """
    start = time.time()
    opt_trainer = OptimizerTrainer(args)
    init_meta_params = opt_trainer.meta_params.detach().cpu().numpy()
    print(f"Initial meta-params: {init_meta_params}")

    metrics = defaultdict(list)
    if args.method == "full":
        assert args.optimizer_name == "LayerSGD"
        return torch.ones(opt_trainer.lopt_info["num_features"]).float(), metrics
    elif args.method == "surgical":
        assert args.optimizer_name == "LayerSGD"
        assert args.layer is not None
        meta_params = (-100 * torch.ones(opt_trainer.lopt_info["num_features"])).float()
        meta_params[2*args.layer] = 100
        meta_params[2*args.layer + 1] = 100
        return meta_params, metrics
    assert args.method in ["ours", "ours-avg"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    meta_learning_info = [
        f"Pre={args.pretrain_dist}, ID={args.ft_id_dist}, OOD={args.ft_ood_dist}, Test={args.test_dist}",
        f"Num meta params: {opt_trainer.meta_params.numel()}",
        f"Outer loop info:",
        f"\tsteps={args.meta_steps}, bs={args.meta_batch_size}, lr={args.meta_lr}, noise_std={args.noise_std}",
        f"Inner loop info:",
        f"\tsteps={args.inner_steps}, bs={args.batch_size}, lr={args.inner_lr}",
        ]
    if args.optimizer_name == "LOptNet":
        meta_learning_info.append(f"LOptNet features: {args.features}")
    print("\n".join(meta_learning_info), "\n")

    for meta_step in range(args.meta_steps + 1):
        if meta_step % args.val_freq == 0:
            val_metrics = opt_trainer.validation(args.val_meta_batch_size)
            for k, v in val_metrics.items():
                metrics[f"{k}_post"].append(np.array(v).mean())
            save_meta_params(opt_trainer, args.exp_name, meta_step)
        _, _, meta_train_loss = opt_trainer.outer_loop_step()
        metrics[f"meta_train_loss"].append(meta_train_loss)

    elapsed = time.time() - start
    print(f"Final meta-params: {opt_trainer.meta_params.detach().cpu().numpy()}")
    print(f"Time taken: {elapsed:.2f}s")
    save_meta_params(opt_trainer, args.exp_name, args.meta_steps)

    meta_params = opt_trainer.meta_params.detach().cpu()
    if args.method == "ours-avg":
        meta_params = torch.sigmoid(meta_params.mean()).repeat(4)
    return meta_params, metrics

def finetune_with_meta_params(meta_params, net, args):
    """ Fine-tune pretrained net with meta-learned optimizer. """
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, output_channels=args.output_channels,
                         num_workers=args.num_workers, use_meta_batch=False)

    if not args.ft_id_ood:  # Fine-tune on ID data only
        train_loader, _ = get_dataloaders(
            dataset_names=[args.ft_id_dist], **loader_kwargs)
    else:  # Fine-tune on both ID and OOD data
        train_loader, _ = get_dataloaders(
            dataset_names=[args.ft_id_dist, args.ft_ood_dist], **loader_kwargs)

    _, src_val_loader = get_dataloaders(dataset_names=[args.pretrain_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = args.id_samples_per_class
    _, id_val_loader = get_dataloaders(dataset_names=[args.ft_id_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = args.ood_samples_per_class
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ft_ood_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = -1
    _, test_loader = get_dataloaders(dataset_names=[args.test_dist], **loader_kwargs)

    optimizer_module = importlib.import_module(f"optimizers.{args.optimizer_name.lower()}")
    optimizer_obj = getattr(optimizer_module, args.optimizer_name)
    ft_net, ft_metrics = train(
        num_epochs=args.num_epochs,
        model=net,
        meta_params=meta_params,
        src_val_loader=src_val_loader,
        train_loader=train_loader,
        id_val_loader=id_val_loader,
        ood_val_loader=ood_val_loader,
        test_loader=test_loader,
        optimizer_obj=optimizer_obj if args.method != "ours-avg" else optimizers.LayerSGD,
        lr=args.inner_lr,
        args=args
    )
    return ft_net, ft_metrics


def evaluate_ft_net(ft_net, args):
    loader_kwargs = dict(root_dir=args.data_dir, output_channels=args.output_channels, batch_size=args.batch_size,
                         num_workers=args.num_workers, use_meta_batch=False)
    _, source_val_loader = get_dataloaders(dataset_names=[args.pretrain_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = args.id_samples_per_class
    _, id_val_loader = get_dataloaders(dataset_names=[args.ft_id_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = args.ood_samples_per_class
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ft_ood_dist], **loader_kwargs)
    loader_kwargs["num_samples_per_class"] = -1
    _, test_loader = get_dataloaders(dataset_names=[args.test_dist], **loader_kwargs)

    metrics = {}
    metrics.update({f"src/{k}": v for k, v in evaluate_net(ft_net, source_val_loader).items()})
    metrics.update({f"id_val/{k}": v for k, v in evaluate_net(ft_net, id_val_loader).items()})
    metrics.update({f"ood_val/{k}": v for k, v in evaluate_net(ft_net, ood_val_loader).items()})
    test_evals = evaluate_net(ft_net, test_loader)
    metrics.update({f"test/{k}": v for k, v in test_evals.items()})

    print(
        f"Source Val Acc: {100 * metrics['src/acc']:.2f} Loss: {metrics['src/loss']:.2f}\n"
        f"ID Val Acc: {100 * metrics['id_val/acc']:.2f} Loss: {metrics['id_val/loss']:.2f}\n"
        f"OOD Val Acc: {100 * metrics['ood_val/acc']:.2f} Loss: {metrics['ood_val/loss']:.2f}\n"
        f"Test Acc: {100 * metrics['test/acc']:.2f} Loss: {metrics['test/loss']:.2f}\n"
    )
    return metrics

def run_method(args):
    pretrain_nets(
        ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, data_dir=args.data_dir,
        output_channels=args.output_channels, num_nets=args.num_nets, num_epochs=args.num_epochs
    )
    print(f"\n--- Method: {args.method} ---\n")

    all_metrics = defaultdict(list)
    final_eval_metrics = defaultdict(list)
    for seed in args.seeds:
        set_seed(seed)

        pretrained_net = copy.deepcopy(get_pretrained_net_fixed(
            ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, output_channels=args.output_channels,
            train=False).to(device))

        if args.method == "pretrained":
            ft_net = pretrained_net
        elif args.method == "lp-ft":
            args.method = "surgical"
            args.layer = -1
            meta_params, _ = train_optimizer(args)
            lp_net, _ = finetune_with_meta_params(meta_params, pretrained_net, args)
            args.method = "full"
            meta_params, _ = train_optimizer(args)
            args.inner_lr *= 1e-1
            ft_net, ft_metrics = finetune_with_meta_params(meta_params, lp_net, args)
        else:
            meta_params, meta_l_metrics = train_optimizer(args)
            ft_net, ft_metrics = finetune_with_meta_params(meta_params, pretrained_net, args)
            for k, v in meta_l_metrics.items():
                all_metrics[f"meta/{k}"].append(v)
            for k, v in ft_metrics.items():
                all_metrics[f"ft/{k}"].append(v)
        eval_metrics = evaluate_ft_net(ft_net, args)
        for k, v in eval_metrics.items():
            final_eval_metrics[k].append(v)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}
    all_metrics = {k: v.mean(0) for k, v in all_metrics.items()}
    final_eval_metrics = {k: np.array(v) for k, v in final_eval_metrics.items()}

    if not args.no_wandb:
        # hack to log all stepwise metrics w/o calling wandb.log twice, so that the x-axes all start from 0,
        # considering different number of steps for meta vs ft metrics
        N = max([len(v) for _, v in all_metrics.items()])
        for i in range(N):
            wandb.log({k: v[i] for k, v in all_metrics.items() if i < len(v)}, step=i)
    print(
        f"\n--- Results (seeds = {args.seeds}) ---\n"
        f"Source Val Acc: {100 * final_eval_metrics['src/acc'].mean():.2f} +- {100 * final_eval_metrics['src/acc'].std():.2f}\n"
        f"ID Val Acc: {100 * final_eval_metrics['id_val/acc'].mean():.2f} +- {100 * final_eval_metrics['id_val/acc'].std():.2f}\n"
        f"OOD Val Acc: {100 * final_eval_metrics['ood_val/acc'].mean():.2f} +- {100 * final_eval_metrics['ood_val/acc'].std():.2f}\n"
        f"Test Acc: {100 * final_eval_metrics['test/acc'].mean():.2f} +- {100 * final_eval_metrics['test/acc'].std():.2f}\n"
        f"\n----------------------------------------\n"
        )
    wandb_dict = {
        "Source Val Accuracy (Avg)": 100 * final_eval_metrics["src/acc"].mean(),
        "Source Val Accuracy (Std)": 100 * final_eval_metrics["src/acc"].std(),
        "ID Val Accuracy (Avg)": 100 * final_eval_metrics["id_val/acc"].mean(),
        "ID Val Accuracy (Std)": 100 * final_eval_metrics["id_val/acc"].std(),
        "OOD Val Accuracy (Avg)": 100 * final_eval_metrics["ood_val/acc"].mean(),
        "OOD Val Accuracy (Std)": 100 * final_eval_metrics["ood_val/acc"].std(),
        "Test Accuracy (Avg)": 100 * final_eval_metrics["test/acc"].mean(),
        "Test Accuracy (Std)": 100 * final_eval_metrics["test/acc"].std(),
    }
    if not args.no_wandb:
        wandb.log(wandb_dict)


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    pickle.dump(args, open(f"results/{args.exp_name}/args.pkl", "wb"))

    # Log hyperparameters and run details to WandB
    if not args.no_wandb:
        config = copy.deepcopy(vars(args))
        ignore_keys = ["ckpt_path", "num_workers", "run_parallel"]
        for key in ignore_keys:
            del config[key]
        wandb.init(project="robust-ft-meta", config=config, name=args.exp_name)

    # Run method
    run_method(args)