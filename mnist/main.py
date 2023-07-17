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

import optimizers
from baselines import evaluate_net, train
from learned_optimizer import OptimizerTrainer
from networks import get_pretrained_net_fixed, pretrain_nets
from mnist import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def save_meta_params(opt_trainer, exp_name: str, meta_step: int):
    meta_params = opt_trainer.meta_params.cpu().detach().numpy()
    fn = f"results/{exp_name}/{meta_step}.npy"
    np.save(fn, np.array(meta_params))
    # print(f"Saved results to {fn}")

def train_optimizer(args):
    """ Train optimizer and return meta-params. """
    if args.method == "full":
        assert args.optimizer_name == "LayerSGD"
        return torch.ones(4).float()
    elif args.method == "surgical":
        assert args.optimizer_name == "LayerSGD"
        return torch.tensor([100, -100, -100, -100]).float()
    assert args.method in ["ours", "ours-avg"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    pretrain_nets(ckpt_path=args.ckpt_path, data_dir=args.data_dir, num_nets=args.num_nets)
    start = time.time()
    opt_trainer = OptimizerTrainer(args)

    meta_learning_info = [
        f"ID={args.ft_id_dist}, OOD={args.ft_ood_dist}, Test={args.test_dist}",
        f"Num meta params: {opt_trainer.meta_params.numel()}",
        f"Outer loop info:",
        f"\tsteps={args.meta_steps}, bs={args.meta_batch_size}, lr={args.meta_lr}, noise_std={args.noise_std}",
        f"Inner loop info:",
        f"\tsteps={args.inner_steps}, bs={args.batch_size}, lr={args.inner_lr}",
        ]
    if args.optimizer_name == "LOptNet":
        meta_learning_info.append(f"LOptNet features: {args.features}")
    print("\n".join(meta_learning_info), "\n")

    metrics = defaultdict(list)
    best_val_loss = np.inf
    for meta_step in range(args.meta_steps + 1):
        if meta_step % args.val_freq == 0:
            losses = opt_trainer.validation(args.val_meta_batch_size)
            for k, v in losses.items():
                metrics[f"{k}_loss_post"].append(np.array(v).mean())
            save_meta_params(opt_trainer, args.exp_name, meta_step)
        opt_trainer.outer_loop_step()

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
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_meta_batch=False)
    # _, source_val_loader = get_dataloaders(dataset_names=["mnist"], **loader_kwargs)

    if not args.ft_id_ood:  # Fine-tune on ID data only
        train_loader, id_val_loader = get_dataloaders(
            dataset_names=[args.ft_id_dist], **loader_kwargs)
        _, ood_val_loader = get_dataloaders(
            dataset_names=[args.ft_ood_dist], **loader_kwargs)
        val_loader = id_val_loader if args.val == "id" else ood_val_loader
    else:  # Fine-tune on both ID and OOD data
        train_loader, val_loader = get_dataloaders(
            dataset_names=[args.ft_id_dist, args.ft_ood_dist], **loader_kwargs)
        if args.val == "id":
            _, val_loader = get_dataloaders(
                dataset_names=[args.ft_ood_dist], **loader_kwargs)

    pretrained_net = copy.deepcopy(get_pretrained_net_fixed(ckpt_path=args.ckpt_path, train=False).to(device))

    optimizer_obj = getattr(optimizers, args.optimizer_name)
    ft_net, train_losses, val_losses = train(
        num_epochs=args.num_epochs,
        model=pretrained_net,
        meta_params=meta_params,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_obj=optimizer_obj if args.method != "ours-avg" else optimizers.LayerSGD,
        lr=args.inner_lr,
        patience=args.patience,
        features=args.features,
        l2_lambda=args.l2_lambda,
    )

    return ft_net, train_losses, val_losses

def evaluate_ft_net(ft_net, args):
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_meta_batch=False)
    _, source_val_loader = get_dataloaders(dataset_names=["mnist"], **loader_kwargs)
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
    print(f"\n--- Method: {args.method} ---\n")
    assert args.method in ["full", "surgical", "ours", "ours-avg"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    all_metrics = defaultdict(list)
    for seed in args.seeds:
        set_seed(seed)

        meta_params, meta_l_metrics = train_optimizer(args)
        ft_net, train_losses, val_losses = finetune_with_meta_params(meta_params, args)
        eval_metrics = evaluate_ft_net(ft_net, args)
        for k, v in meta_l_metrics.items():
            all_metrics[f"meta/{k}"].append(v)
        for k, v in eval_metrics.items():
            all_metrics[k].append(v)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    if not args.no_wandb:
        for train_loss, val_loss in zip(train_losses, val_losses):
            # Log train and val losses of the last run (diff lengths of training due to early stopping)
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

        # Log average meta losses per meta-step
        avg_meta_losses = {k: v.mean(0) for k, v in all_metrics.items() if k.startswith("meta/")}
        N = len(meta_l_metrics["source_loss_post"])
        for i in range(N):
            wandb.log({k: v[i] for k, v in avg_meta_losses.items()})

    print(
        f"\n--- Results (seeds = {args.seeds}) ---\n"
        f"Source Val Acc: {100 * all_metrics['src/acc'].mean():.2f} +- {100 * all_metrics['src/acc'].std():.2f}\n"
        f"ID Val Acc: {100 * all_metrics['id_val/acc'].mean():.2f} +- {100 * all_metrics['id_val/acc'].std():.2f}\n"
        f"OOD Val Acc: {100 * all_metrics['ood_val/acc'].mean():.2f} +- {100 * all_metrics['ood_val/acc'].std():.2f}\n"
        f"Test Acc: {100 * all_metrics['test/acc'].mean():.2f} +- {100 * all_metrics['test/acc'].std():.2f}\n"
        f"\n----------------------------------------\n"
        )
    wandb_dict = {
        "Source Val Accuracy (Avg)": 100 * all_metrics["src/acc"].mean(),
        "Source Val Accuracy (Std)": 100 * all_metrics["src/acc"].std(),
        "ID Val Accuracy (Avg)": 100 * all_metrics["id_val/acc"].mean(),
        "ID Val Accuracy (Std)": 100 * all_metrics["id_val/acc"].std(),
        "OOD Val Accuracy (Avg)": 100 * all_metrics["ood_val/acc"].mean(),
        "OOD Val Accuracy (Std)": 100 * all_metrics["ood_val/acc"].std(),
        "Test Accuracy (Avg)": 100 * all_metrics["test/acc"].mean(),
        "Test Accuracy (Std)": 100 * all_metrics["test/acc"].std(),
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