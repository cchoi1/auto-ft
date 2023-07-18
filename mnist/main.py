import argparse
import copy
import importlib
import os
import pickle
import random
import time

import numpy as np
import torch
import wandb

from datasets import _CORRUPTIONS
from datasets import get_dataloaders
from learned_optimizer import OptimizerTrainer
from networks import get_pretrained_net_fixed
from networks import pretrain_nets
from utils import save_meta_params, set_seed, train, evaluate_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_lopt(args):
    start = time.time()
    opt_trainer = OptimizerTrainer(args)
    for meta_step in range(args.meta_steps + 1):
        if meta_step % args.val_freq == 0:
            losses = opt_trainer.validation(args.val_meta_batch_size)
            save_meta_params(opt_trainer, args.exp_name, meta_step)
        # outer_start = time.time()
        opt_trainer.outer_loop_step()
        # print(f"Outer Loop Time: {time.time() - outer_start:.2f}")

    elapsed = time.time() - start
    print(f"Final meta-params: {opt_trainer.meta_params.detach().cpu().numpy()}")
    print(f"Time taken: {elapsed:.2f}s")
    save_meta_params(opt_trainer, args.exp_name, args.meta_steps)

    return opt_trainer.meta_params


def get_ft_net(method, args):
    pretrained_net = copy.deepcopy(get_pretrained_net_fixed(ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, train=False).to(device))
    num_params = len([p for p in pretrained_net.parameters()])
    if method == "full":
        meta_params = torch.ones(num_params).float()
    elif method == "surgical":
        meta_params = torch.full((num_params,), -100)
        meta_params[0] = 100
        meta_params = meta_params.float()
    elif "ours" in method:
        meta_params = train_lopt(args)
    else:
        raise ValueError("Method must be 'full', 'surgical', 'ours', 'ours-avg")

    _, source_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=["mnist"], batch_size=args.batch_size,
                                           meta_batch_size=args.meta_batch_size, num_workers=args.num_workers,
                                           use_meta_batch=False)
    if not args.ft_id_ood:
        train_loader, id_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_id_dist],
                                                      batch_size=args.batch_size,
                                                      meta_batch_size=args.meta_batch_size // 2,
                                                      num_workers=args.num_workers, use_meta_batch=False)
        _, ood_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
                                            batch_size=args.batch_size,
                                            meta_batch_size=args.meta_batch_size // 2,
                                            num_workers=args.num_workers, use_meta_batch=False)
        val_loader = id_val_loader if args.val == "id" else ood_val_loader
    else:
        """Fine-tune on both ID and OOD data."""
        train_loader, val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_id_dist, args.ft_ood_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
        if args.val == "id":
            _, val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
    if method == "ours-avg":
        meta_params = torch.sigmoid(meta_params.mean()).repeat(num_params)

    optimizer_module = importlib.import_module(f"optimizers")
    optimizer_obj = getattr(optimizer_module, args.optimizer_name)
    ft_net, train_losses, val_losses = train(
        num_epochs=args.num_epochs,
        model=pretrained_net,
        meta_params=meta_params,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_obj=optimizer_obj,
        lr=0.1,
        patience=args.patience,
        l2_lambda=args.l2_lambda,
    )

    return ft_net, train_losses, val_losses


def evaluate_ft_net(ft_net, args):
    _, source_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.pretrain_dist], batch_size=args.batch_size,
                                           meta_batch_size=args.meta_batch_size, num_workers=args.num_workers,
                                           use_meta_batch=False)
    _, id_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_id_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
    _, ood_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
    _, test_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.test_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
    source_val_accs, source_val_losses = [], []
    id_val_accs, id_val_losses = [], []
    ood_val_accs, ood_val_losses = [], []
    test_accs, test_losses = [], []

    # Get accuracy and loss on source distribution.
    acc, losses = evaluate_net(ft_net, source_val_loader)
    source_val_accs.append(acc)
    source_val_losses.append(losses[-1])
    # Get accuracy and loss on ID and OOD ft distributions.
    acc, losses = evaluate_net(ft_net, id_val_loader)
    id_val_accs.append(acc)
    id_val_losses.append(losses[-1])
    acc, losses = evaluate_net(ft_net, ood_val_loader)
    ood_val_accs.append(acc)
    ood_val_losses.append(losses[-1])
    # Get accuracy and loss on test distribution.
    acc, losses = evaluate_net(ft_net, test_loader)
    test_accs.append(acc)
    test_losses.append(losses[-1])

    source_val_accs, source_val_losses = np.array(source_val_accs), np.array(source_val_losses)
    id_val_accs, id_val_losses = np.array(id_val_accs), np.array(id_val_losses)
    ood_val_accs, ood_val_losses = np.array(ood_val_accs), np.array(ood_val_losses)
    test_accs, test_losses = np.array(test_accs), np.array(test_losses)
    print(
        f"Source Val Acc: {100 * source_val_accs.mean():.2f} +- {100 * source_val_accs.std():.2f} "
        f"| Source Val Loss: {source_val_losses.mean():.2f} +- {source_val_losses.std():.2f}")
    print(
        f"ID Val Acc: {100 * id_val_accs.mean():.2f} +- {100 * id_val_accs.std():.2f} "
        f"| ID Val Loss: {id_val_losses.mean():.2f} +- {id_val_losses.std():.2f}")
    print(
        f"OOD Val Acc: {100 * ood_val_accs.mean():.2f} +- {100 * ood_val_accs.std():.2f} "
        f"| OOD Val Loss: {ood_val_losses.mean():.2f} +- {ood_val_losses.std():.2f}")
    print(
        f"Test Acc: {100 * test_accs.mean():.2f} +- {100 * test_accs.std():.2f} "
        f"| Test Loss: {test_losses.mean():.2f} +- {test_losses.std():.2f}")
    return source_val_accs.mean(), id_val_accs.mean(), ood_val_accs.mean(), test_accs.mean()


def run_method(method, args):
    print(f"\n--- Method: {method} ---\n")
    assert method in ["full", "surgical", "ours", "ours-avg"], \
        "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    pretrain_nets(ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, data_dir=args.data_dir,
                  num_nets=args.num_nets, num_epochs=args.num_epochs)
    source_val_accs, id_val_accs, ood_val_accs, test_accs = [], [], [], []
    for seed in args.seeds:
        set_seed(seed)
        ft_net, train_losses, val_losses = get_ft_net(method, args)

        source_val_acc, id_val_acc, ood_val_acc, test_acc = evaluate_ft_net(ft_net, args)
        source_val_accs.append(source_val_acc)
        id_val_accs.append(id_val_acc)
        ood_val_accs.append(ood_val_acc)
        test_accs.append(test_acc)

    # Log train and val losses at each epoch of the last run (diff lengths of training due to early stopping)
    for train_loss, val_loss in zip(train_losses, val_losses):
        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

    # Log final accuracies
    source_val_accs, id_val_accs, ood_val_accs, test_accs = np.array(source_val_accs), np.array(id_val_accs), np.array(ood_val_accs), np.array(test_accs)
    print(f"\n--- Results (seeds = {args.seeds}) ---\n")
    print(f"Source Val Acc: {100 * source_val_accs.mean():.2f} +- {100 * source_val_accs.std():.2f}")
    print(f"ID Val Acc: {100 * id_val_accs.mean():.2f} +- {100 * id_val_accs.std():.2f}")
    print(f"OOD Val Acc: {100 * ood_val_accs.mean():.2f} +- {100 * ood_val_accs.std():.2f}")
    print(f"Test Acc: {100 * test_accs.mean():.2f} +- {100 * test_accs.std():.2f}")
    print(f"\n----------------------------------------\n")
    wandb.log({
        "Source Val Accuracy (Avg)": 100 * source_val_accs.mean(),
        "Source Val Accuracy (Std)": 100 * source_val_accs.std(),
        "ID Val Accuracy (Avg)": 100 * id_val_accs.mean(),
        "ID Val Accuracy (Std)": 100 * id_val_accs.std(),
        "OOD Val Accuracy (Avg)": 100 * ood_val_accs.mean(),
        "OOD Val Accuracy (Std)": 100 * ood_val_accs.std(),
        "Test Accuracy (Avg)": 100 * test_accs.mean(),
        "Test Accuracy (Std)": 100 * test_accs.std(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dists = ["mnist", "mnistc", "mnist-label-shift", "svhn"] + _CORRUPTIONS
    parser.add_argument("--method", type=str, choices=["full", "surgical", "ours", "ours-avg"])
    parser.add_argument(
        "--pretrain_dist",
        type=str,
        default="svhn",
        choices=dists,
    )
    parser.add_argument(
        "--ft_id_dist",
        type=str,
        default="mnistc",
        choices=dists,
    )
    parser.add_argument(
        "--ft_ood_dist",
        type=str,
        default="mnistc",
        choices=dists,
    )
    parser.add_argument(
        "--test_dist",
        type=str,
        default="mnist-label-shift",
        choices=dists,
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="LayerSGD",
    )
    parser.add_argument("--ft_id_ood", action="store_true", help="Fine-tune w/ meta-params on both ID and OOD data.")

    parser.add_argument("--features", nargs='+', type=str,
                        help="Choose a subset of [p, g, p_norm, g_norm, g_norm_avg, depth, dist_init_param, iter, loss, loss_ema, tensor_rank].",
                        default=None)
    parser.add_argument("--meta_steps", type=int)
    parser.add_argument("--inner_steps", type=int)
    parser.add_argument("--meta_loss_avg_w", type=float, default=0.0)
    parser.add_argument("--meta_loss_final_w", type=float, default=1.0)
    parser.add_argument("--meta_batch_size", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--val_meta_batch_size", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_nets", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--l2_lambda", type=float, default=None)
    parser.add_argument("--val", type=str, choices='id, ood')
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])

    # Dataset & Dataloader
    parser.add_argument(
        "--data_dir", type=str, default="/iris/u/cchoi1/Data"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--ckpt_path", type=str, default="/iris/u/cchoi1/robust-optimizer/mnist/ckpts"
    )

    # Parallelize inner loop
    parser.add_argument("--run_parallel", action="store_true")

    args = parser.parse_args()
    print(args)

    defaults = vars(parser.parse_args([]))
    actuals = vars(args)
    nondefaults = []
    for k in sorted(actuals.keys()):
        if defaults[k] != actuals[k]:
            first_letters = "".join([w[0] for w in k.split("_")])
            if type(actuals[k]) == float:
                nondefaults.append(f"{first_letters}={actuals[k]:.2e}")
            else:
                nondefaults.append(f"{first_letters}={actuals[k]}")
    if len(nondefaults) > 0:
        args.exp_name = "_".join(nondefaults)
    else:
        args.exp_name = "default"
    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    pickle.dump(args, open(f"results/{args.exp_name}/args.pkl", "wb"))

    wandb.init(mode='disabled')
    # Log hyperparameters and run details to WandB
    # config = copy.deepcopy(vars(args))
    # del config["data_dir"], config["num_workers"], config["ckpt_path"], config["run_parallel"]
    # wandb.init(
    #     project="robust-ft-meta",
    #     config=config,
    #     name=args.exp_name,
    # )

    # Run method
    run_method(args.method, args)