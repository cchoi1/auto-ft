import copy
import os
import pickle
import random
import time
from collections import defaultdict
from parser import get_args

import numpy as np
import torch
import wandb

from optimizers import optimizers
from utils import evaluate_net, train, save_meta_params, set_seed
from learned_optimizer import OptimizerTrainer
from networks import get_pretrained_net_fixed, pretrain_nets
from data.dataloaders import get_dataloaders

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
        meta_params = (-100 * torch.ones(opt_trainer.lopt_info["num_features"])).float()
        #TODO (cchoi1) - add support for surgical fine-tuning an arbitrary network
        if args.pretrain_dist == "mnist":
            meta_params[0] = 100
        elif args.pretrain_dist == "svhn":
            meta_params[0] = 100
            meta_params[1] = 100
        else:
            raise ValueError(f"Unsupported pretrain_dist {args.pretrain_dist}")
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

def finetune_with_meta_params(meta_params, args):
    """ Fine-tune pretrained net with meta-learned optimizer. """
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, output_channels=args.output_channels, num_workers=args.num_workers, use_meta_batch=False)

    if not args.ft_id_ood:  # Fine-tune on ID data only
        train_loader, _ = get_dataloaders(
            dataset_names=[args.ft_id_dist], **loader_kwargs)
    else:  # Fine-tune on both ID and OOD data
        train_loader, _ = get_dataloaders(
            dataset_names=[args.ft_id_dist, args.ft_ood_dist], **loader_kwargs)

    _, src_val_loader = get_dataloaders(dataset_names=[args.pretrain_dist], **loader_kwargs)
    _, id_val_loader = get_dataloaders(dataset_names=[args.ft_id_dist], **loader_kwargs)
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ft_ood_dist], **loader_kwargs)
    _, test_loader = get_dataloaders(dataset_names=[args.test_dist], **loader_kwargs)

    pretrained_net = copy.deepcopy(get_pretrained_net_fixed(
        ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, output_channels=args.output_channels,
        train=False).to(device))

    optimizer_obj = getattr(optimizers, args.optimizer_name)
    train_kwargs = {"val": args.val, "lr": args.inner_lr, "patience": args.patience, "features": args.features,
                    "lopt_net_dim": args.lopt_net_dim, "l2_lambda": args.l2_lambda, "wnb": args.wnb}
    ft_net, ft_metrics = train(
        num_epochs=args.num_epochs,
        model=pretrained_net,
        meta_params=meta_params,
        src_val_loader=src_val_loader,
        train_loader=train_loader,
        id_val_loader=id_val_loader,
        ood_val_loader=ood_val_loader,
        test_loader=test_loader,
        optimizer_obj=optimizer_obj if args.method != "ours-avg" else optimizers.LayerSGD,
        **train_kwargs
    )
    return ft_net, ft_metrics


def evaluate_ft_net(ft_net, args):
    loader_kwargs = dict(root_dir=args.data_dir, output_channels=args.output_channels, batch_size=args.batch_size, num_workers=args.num_workers, use_meta_batch=False)
    _, source_val_loader = get_dataloaders(dataset_names=[args.pretrain_dist], **loader_kwargs)
    _, id_val_loader = get_dataloaders(dataset_names=[args.ft_id_dist], **loader_kwargs)
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ft_ood_dist], **loader_kwargs)
    _, test_loader = get_dataloaders(dataset_names=[args.test_dist], **loader_kwargs)

    metrics = {}
    metrics.update({f"src/{k}": v for k, v in evaluate_net(ft_net, source_val_loader).items()})
    metrics.update({f"id_val/{k}": v for k, v in evaluate_net(ft_net, id_val_loader).items()})
    metrics.update({f"ood_val/{k}": v for k, v in evaluate_net(ft_net, ood_val_loader).items()})
    metrics.update({f"test/{k}": v for k, v in evaluate_net(ft_net, test_loader).items()})

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
    assert args.method in ["full", "surgical", "ours", "ours-avg", "pretrained"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    all_metrics = defaultdict(list)
    final_eval_metrics = defaultdict(list)
    for seed in args.seeds:
        set_seed(seed)

        if args.method == "pretrained":
            ft_net = copy.deepcopy(get_pretrained_net_fixed(
                ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, output_channels=args.output_channels,
                train=False).to(device))
        else:
            meta_params, meta_l_metrics = train_optimizer(args)
            ft_net, ft_metrics = finetune_with_meta_params(meta_params, args)
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
        N = max(len(all_metrics["meta/id_loss_post"]), len(ft_metrics["train_loss"]), len(all_metrics["meta/meta_train_loss"]))
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