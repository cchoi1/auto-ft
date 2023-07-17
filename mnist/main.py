import copy
import os
import pickle
import random
import time

import numpy as np
import optimizers
import torch
import wandb
from baselines import evaluate_net, train
from learned_optimizer import OptimizerTrainer
from networks import get_pretrained_net_fixed, pretrain_nets

from parser import get_args
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
        return torch.ones(4).float()
    elif args.method == "surgical":
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

    best_val_loss = np.inf
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

    meta_params = opt_trainer.meta_params.detach().cpu()
    if args.method == "ours-avg":
        meta_params = torch.sigmoid(meta_params.mean()).repeat(4)
    return meta_params

def get_ft_net(meta_params, args):
    """ Fine-tune pretrained net with meta-learned optimizer. """
    loader_kwargs = dict(root_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_meta_batch=False)
    _, source_val_loader = get_dataloaders(dataset_names=["mnist"], **loader_kwargs)

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
        optimizer_obj=optimizer_obj,
        lr=args.inner_lr,
        patience=args.patience,
        features=args.features,
        l2_lambda=args.l2_lambda,
    )

    return ft_net, train_losses, val_losses

def evaluate_ft_net(ft_net, args):
    _, source_val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=["mnist"], batch_size=args.batch_size,
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

def run_method(args):
    print(f"\n--- Method: {args.method} ---\n")
    assert args.method in ["full", "surgical", "ours", "ours-avg"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    source_val_accs, id_val_accs, ood_val_accs, test_accs = [], [], [], []
    for seed in args.seeds:
        set_seed(seed)

        meta_params = train_optimizer(args)
        ft_net, train_losses, val_losses = get_ft_net(meta_params, args)
        source_val_acc, id_val_acc, ood_val_acc, test_acc = evaluate_ft_net(ft_net, args)

        source_val_accs.append(source_val_acc)
        id_val_accs.append(id_val_acc)
        ood_val_accs.append(ood_val_acc)
        test_accs.append(test_acc)

    # Log train and val losses at each epoch of the last run (diff lengths of training due to early stopping)
    if not args.no_wandb:
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
    wandb_dict = {
        "Source Val Accuracy (Avg)": 100 * source_val_accs.mean(),
        "Source Val Accuracy (Std)": 100 * source_val_accs.std(),
        "ID Val Accuracy (Avg)": 100 * id_val_accs.mean(),
        "ID Val Accuracy (Std)": 100 * id_val_accs.std(),
        "OOD Val Accuracy (Avg)": 100 * ood_val_accs.mean(),
        "OOD Val Accuracy (Std)": 100 * ood_val_accs.std(),
        "Test Accuracy (Avg)": 100 * test_accs.mean(),
        "Test Accuracy (Std)": 100 * test_accs.std(),
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