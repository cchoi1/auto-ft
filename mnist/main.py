import copy
import importlib
import os
import pickle
import random
import time
from collections import defaultdict
from parser import get_args

import numpy as np
import torch
import wandb

<<<<<<< HEAD
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
=======
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
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
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

<<<<<<< HEAD
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
=======
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
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
        val_loader = id_val_loader if args.val == "id" else ood_val_loader
    else:  # Fine-tune on both ID and OOD data
        train_loader, val_loader = get_dataloaders(
            dataset_names=[args.ft_id_dist, args.ft_ood_dist], **loader_kwargs)
        if args.val == "id":
<<<<<<< HEAD
            _, val_loader = get_dataloaders(root_dir=args.data_dir, dataset_names=[args.ft_ood_dist],
                                                  batch_size=args.batch_size, meta_batch_size=args.meta_batch_size // 2,
                                                  num_workers=args.num_workers, use_meta_batch=False)
    if method == "ours-avg":
        meta_params = torch.sigmoid(meta_params.mean()).repeat(num_params)

    optimizer_module = importlib.import_module(f"optimizers")
    optimizer_obj = getattr(optimizer_module, args.optimizer_name)
=======
            _, val_loader = get_dataloaders(
                dataset_names=[args.ft_ood_dist], **loader_kwargs)

    pretrained_net = copy.deepcopy(get_pretrained_net_fixed(ckpt_path=args.ckpt_path, train=False).to(device))

    optimizer_obj = getattr(optimizers, args.optimizer_name)
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
    ft_net, train_losses, val_losses = train(
        num_epochs=args.num_epochs,
        model=pretrained_net,
        meta_params=meta_params,
        train_loader=train_loader,
        val_loader=val_loader,
<<<<<<< HEAD
        optimizer_obj=optimizer_obj,
        lr=0.1,
=======
        optimizer_obj=optimizer_obj if args.method != "ours-avg" else optimizers.LayerSGD,
        lr=args.inner_lr,
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
        patience=args.patience,
        features=args.features,
        l2_lambda=args.l2_lambda,
    )

    return ft_net, train_losses, val_losses


def evaluate_ft_net(ft_net, args):
<<<<<<< HEAD
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
=======
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
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640

    print(
        f"Source Val Acc: {100 * metrics['src/acc']:.2f} Loss: {metrics['src/loss']:.2f}\n"
        f"ID Val Acc: {100 * metrics['id_val/acc']:.2f} Loss: {metrics['id_val/loss']:.2f}\n"
        f"OOD Val Acc: {100 * metrics['ood_val/acc']:.2f} Loss: {metrics['ood_val/loss']:.2f}\n"
        f"Test Acc: {100 * metrics['test/acc']:.2f} Loss: {metrics['test/loss']:.2f}\n"
    )
    return metrics

<<<<<<< HEAD

def run_method(method, args):
    print(f"\n--- Method: {method} ---\n")
    assert method in ["full", "surgical", "ours", "ours-avg"], \
        "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    pretrain_nets(ckpt_path=args.ckpt_path, dataset_name=args.pretrain_dist, data_dir=args.data_dir,
                  num_nets=args.num_nets, num_epochs=args.num_epochs)
    source_val_accs, id_val_accs, ood_val_accs, test_accs = [], [], [], []
=======
def run_method(args):
    print(f"\n--- Method: {args.method} ---\n")
    assert args.method in ["full", "surgical", "ours", "ours-avg"], "Method must be 'full', 'surgical', 'ours', or 'ours-avg'."

    all_metrics = defaultdict(list)
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
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
<<<<<<< HEAD
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
=======
    args = get_args()
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640
    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    pickle.dump(args, open(f"results/{args.exp_name}/args.pkl", "wb"))

    wandb.init(mode='disabled')
    # Log hyperparameters and run details to WandB
<<<<<<< HEAD
    # config = copy.deepcopy(vars(args))
    # del config["data_dir"], config["num_workers"], config["ckpt_path"], config["run_parallel"]
    # wandb.init(
    #     project="robust-ft-meta",
    #     config=config,
    #     name=args.exp_name,
    # )
=======
    if not args.no_wandb:
        config = copy.deepcopy(vars(args))
        ignore_keys = ["ckpt_path", "num_workers", "run_parallel"]
        for key in ignore_keys:
            del config[key]
        wandb.init(project="robust-ft-meta", config=config, name=args.exp_name)
>>>>>>> 42ddc76f68b729da1eed44383318ff5b29ece640

    # Run method
    run_method(args)