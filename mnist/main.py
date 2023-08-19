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
from utils import evaluate_net, train, save_meta_params, set_seed, get_lloss_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_optimizer(args, method):
    """ Train optimizer and return meta-params. """
    start = time.time()
    opt_trainer = OptimizerTrainer(args)
    init_meta_params = opt_trainer.meta_params.detach().cpu().numpy()
    print(f"Initial meta-params: {init_meta_params}")

    metrics = defaultdict(list)
    if method in ["full", "wise-ft"]:
        # assert args.optimizer_name == "LayerSGD"
        return 100 * torch.ones(opt_trainer.lopt_info["input_dim"]).float(), metrics
    elif method == "surgical":
        assert args.optimizer_name == "LayerSGD"
        assert args.layer is not None
        meta_params = (-100 * torch.ones(opt_trainer.lopt_info["input_dim"])).float()
        meta_params[2*args.layer] = 100
        meta_params[2*args.layer + 1] = 100
        return meta_params, metrics

    meta_learning_info = [
        f"Pre={args.pretrain}, ID={args.id}, OOD={args.ood}, Test={args.test}",
        f"Num meta params: {opt_trainer.meta_params.numel()}",
        f"Outer loop info:",
        f"\tsteps={args.meta_steps}, bs={args.meta_batch_size}, lr={args.meta_lr}, noise_std={args.noise_std}",
        f"Inner loop info:",
        f"\tsteps={args.inner_steps}, bs={args.batch_size}, lr={args.inner_lr}",
        ]
    if args.optimizer_name == "LOptNet":
        meta_learning_info.append(f"LOptNet features: {args.features}")
    print("\n".join(meta_learning_info), "\n")

    num_meta_steps = 1 if args.use_hyperopt else args.meta_steps
    for meta_step in range(num_meta_steps):
        if meta_step % args.val_freq == 1:
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
    if method == "ours-avg":
        meta_params = torch.sigmoid(meta_params.mean()).repeat(4)
    return meta_params, metrics


def finetune_with_meta_params(meta_params, net, args, ft_dist, val):
    """ Fine-tune pretrained net with meta-learned optimizer. """
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, output_channels=args.output_channels,
                         num_workers=args.num_workers, use_meta_batch=False)
    loader_kwargs["num_samples_per_class"] = [-1]

    if ft_dist == "id":  # Fine-tune on ID data only
        train_loader, _ = get_dataloaders(dataset_names=[args.id], **loader_kwargs)
    elif ft_dist == "ood":  # Fine-tune on OOD data only
        train_loader, _ = get_dataloaders(dataset_names=[args.ood], **loader_kwargs)
    elif ft_dist == "id+ood":
        loader_kwargs["num_samples_per_class"] = [-1, -1]
        train_loader, _ = get_dataloaders(dataset_names=[args.id, args.ood], **loader_kwargs)
    elif ft_dist == "src+id":
        loader_kwargs["num_samples_per_class"] = [-1, -1]
        train_loader, _ = get_dataloaders(dataset_names=[args.pretrain, args.id], **loader_kwargs)
    elif ft_dist == "src+ood":
        loader_kwargs["num_samples_per_class"] = [-1, -1]
        train_loader, _ = get_dataloaders(dataset_names=[args.pretrain, args.ood], **loader_kwargs)
    elif ft_dist == "src+id+ood":
        loader_kwargs["num_samples_per_class"] = [-1, -1, -1]
        train_loader, _ = get_dataloaders(dataset_names=[args.pretrain, args.id, args.ood], **loader_kwargs)

    _, src_val_loader = get_dataloaders(dataset_names=[args.pretrain], **loader_kwargs)
    _, id_val_loader = get_dataloaders(dataset_names=[args.id], **loader_kwargs)
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ood], **loader_kwargs)
    _, test_loader = get_dataloaders(dataset_names=[args.test], **loader_kwargs)

    if args.loss_name is not None:
        loss_module = importlib.import_module(f"losses.{args.loss_name.lower()}")
        lloss_info = get_lloss_info(net, args)
        lloss_info['meta_params'] = {'start': 0, 'end': 71}
        loss_fn = getattr(loss_module, args.loss_name)(meta_params, net, lloss_info)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_obj = None
    if args.optimizer_name is not None:
        optimizer_module = importlib.import_module(f"optimizers.{args.optimizer_name.lower()}")
        optimizer_obj = getattr(optimizer_module, args.optimizer_name)
    ft_lr = args.inner_lr if args.ft_lr is None else args.ft_lr
    ft_net, ft_metrics = train(
        num_epochs=args.num_epochs,
        model=net,
        meta_params=meta_params,
        src_val_loader=src_val_loader,
        train_loader=train_loader,
        id_val_loader=id_val_loader,
        ood_val_loader=ood_val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer_obj=optimizer_obj if args.method != "ours-avg" else optimizers.LayerSGD,
        lr=args.inner_lr,
        val=val,
        alpha=args.alpha,
        args=args
    )
    return ft_net, ft_metrics


def evaluate_ft_net(ft_net, args):
    loader_kwargs = dict(root_dir=args.data_dir, output_channels=args.output_channels, batch_size=args.batch_size,
                         num_workers=args.num_workers, use_meta_batch=False)
    loader_kwargs["num_samples_per_class"] = [-1]
    _, source_val_loader = get_dataloaders(dataset_names=[args.pretrain], **loader_kwargs)
    _, id_val_loader = get_dataloaders(dataset_names=[args.id], **loader_kwargs)
    _, ood_val_loader = get_dataloaders(dataset_names=[args.ood], **loader_kwargs)
    _, test_loader = get_dataloaders(dataset_names=[args.test], **loader_kwargs)

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
    assert len(args.method) <= 2, "Must specify at most 2 methods to run sequentially -- on ID, then OOD."
    pretrain_nets(
        ckpt_path=args.ckpt_path, dataset_name=args.pretrain, data_dir=args.data_dir,
        output_channels=args.output_channels, num_nets=args.num_nets, num_epochs=args.num_epochs
    )
    print(f"\n--- Method: {args.method} ---\n")

    all_metrics = defaultdict(list)
    final_eval_metrics = defaultdict(list)
    for seed in args.seeds:
        set_seed(seed)

        pretrained_net = copy.deepcopy(get_pretrained_net_fixed(
            ckpt_path=args.ckpt_path, dataset_name=args.pretrain, output_channels=args.output_channels,
            train=False).to(device))

        for i, method in enumerate(args.method):
            if method == "pretrained":
                ft_net = pretrained_net
            elif method == "lp-ft":
                args.layer = -1
                meta_params, _ = train_optimizer(args, "surgical")
                lp_net, _ = finetune_with_meta_params(meta_params, pretrained_net, args, ft_dist=args.ft_dists[i], val=args.val[i])
                meta_params, meta_l_metrics = train_optimizer(args, "full")
                args.inner_lr *= 1e-1
                ft_net, ft_metrics = finetune_with_meta_params(meta_params, lp_net, args, ft_dist=args.ft_dists[i], val=args.val[i])
                for k, v in meta_l_metrics.items():
                    all_metrics[f"meta/{k}"].append(v)
                for k, v in ft_metrics.items():
                    all_metrics[f"ft/{k}"].append(v)
            else:
                meta_params, meta_l_metrics = train_optimizer(args, method)
                ft_net, ft_metrics = finetune_with_meta_params(meta_params, pretrained_net, args, ft_dist=args.ft_dists[i], val=args.val[i])
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
        wandb.init(project="robust-ft-meta", entity=args.wandb_entity, config=config, name=args.exp_name)

    # Run method
    run_method(args)