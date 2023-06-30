import argparse
import os
import pickle
import time

import numpy as np
import torch

from baselines import fine_tune_epoch, evaluate_net
from learned_optimizer import OptimizerTrainer
from mnist import _CORRUPTIONS
from networks import pretrain_nets

import optimizers
from mnist import load_dataset
from networks import get_pretrained_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def train_lopt(args):
    pretrain_nets(ckpt_path=args.ckpt_path, data_dir=args.data_dir, num_nets=args.num_nets)

    start = time.time()
    opt_trainer = OptimizerTrainer(args)
    for meta_step in range(args.meta_steps + 1):
        if meta_step % args.val_freq == 0:
            opt_trainer.validation(repeat=args.val_meta_batch_size)
            meta_params = opt_trainer.meta_params
            fn = f"results/{args.exp_name}/{meta_step}.npy"
            np.save(fn, np.array(meta_params))
            print(f"Saved results to {fn}")
        # outer_start = time.time()
        opt_trainer.outer_loop_step()
        # print(f"Outer Loop Time: {time.time() - outer_start:.2f}")

    elapsed = time.time() - start
    print(f"Final meta-params: {meta_params}")
    print(f"Time taken: {elapsed:.2f}s")

    meta_params = opt_trainer.meta_params
    fn = f"results/{args.exp_name}/final.npy"
    np.save(fn, np.array(meta_params))
    print(f"Saved results to {fn}")

    return meta_params


def run_method(method, args, meta_params=None):
    print(f"\n--- Method: {method} ---\n")
    if method == "full":
        meta_params = torch.ones(4).float()
    elif method == "surgical":
        meta_params = torch.tensor([100, -100, -100, -100]).float()
    elif method == "ours":
        assert meta_params is not None, "meta_params must be provided for method = 'ours'"
    else:
        raise ValueError("Baseline must be 'full', 'surgical', or 'ours'")

    val_losses, test_losses = [], []
    val_accs, test_accs = [], []
    train_loader, id_val_loader = load_dataset(root_dir=args.data_dir, dataset=args.ft_distribution)
    test_loader, _ = load_dataset(root_dir=args.data_dir, dataset=args.test_distribution)
    for _ in range(args.num_nets):
        _net = get_pretrained_net(ckpt_path=args.ckpt_path, train=False)
        net, meta_params = fine_tune_epoch(
            _net,
            meta_params,
            train_loader,
            optimizers.LayerSGD,
            inner_lr=0.1,
        )
        # Get accuracy and loss on ft distribution.
        acc, losses = evaluate_net(net, id_val_loader)
        val_accs.append(acc)
        val_losses.append(losses[-1])
        # Get accuracy and loss on test distribution.
        acc, losses = evaluate_net(net, test_loader)
        test_accs.append(acc)
        test_losses.append(losses[-1])

    val_losses, test_losses = np.array(val_losses), np.array(test_losses)
    val_accs, test_accs = np.array(val_accs), np.array(test_accs)
    print(
        f"Val Acc: {val_accs.mean():.4f} +- {val_accs.std():.4f} | Val Loss: {val_losses.mean():.4f} +- {val_losses.std():.4f}")
    print(
        f"Test Acc: {test_accs.mean():.4f} +- {test_accs.std():.4f} | Test Loss: {test_losses.mean():.4f} +- {test_losses.std():.4f}")


def sanity_check(args):
    """Sanity check with full vs surgical fine-tuning."""
    for baseline in ["full", "surgical"]:
        run_method(baseline, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dists = ["mnist", "mnistc", "mnist-label-shift"] + _CORRUPTIONS
    parser.add_argument(
        "--ft_distribution",
        type=str,
        default="mnistc",
        choices=dists,
    )
    parser.add_argument(
        "--test_distribution",
        type=str,
        default="mnist-label-shift",
        choices=dists,
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="LayerSGDLinear",
    )
    parser.add_argument("--meta_steps", type=int, default=100)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--meta_loss_avg_w", type=float, default=0.0)
    parser.add_argument("--meta_loss_final_w", type=float, default=1.0)
    parser.add_argument("--meta_batch_size", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--val_meta_batch_size", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--train_N", type=int, default=10)
    parser.add_argument(
        "--data_dir", type=str, default="/afs/cs/u/cchoi1/scr/data"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="/afs/cs/u/cchoi1/scr/robust-optimizer/mnist/ckpts"
    )
    parser.add_argument("--num_nets", type=int, default=3)
    args = parser.parse_args()

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

    sanity_check(args)
    meta_params = train_lopt(args)
    run_method("ours", args, meta_params=meta_params)
