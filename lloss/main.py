import argparse
import copy
import time

import numpy as np
import torch
from clip import clip
from hyperopt import fmin, hp, tpe, Trials
from torch.utils.data import DataLoader

from losses.layerloss import LayerLoss
from data.dataloaders import get_all_datasets, get_subset, get_transform
from networks import get_pretrained_net_fixed
from utils import set_seed, evaluate, finetune
from plots import plot_accuracies


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LOSSES = 7

def get_network(in_dist: str):
    if in_dist == "mnist":
        net = get_pretrained_net_fixed(
            ckpt_path="/iris/u/cchoi1/robust-optimizer/mnist/ckpts",
            dataset_name="svhn",
            output_channels=3,
            train=True,
        )
        transform = None
    elif in_dist == "cifar10":
        model, transform = clip.load("RN50", device=device)
        net = model.visual
    elif in_dist == "imagenet":
        model, transform = clip.load("ViT-B/32", device=device)
        net = model.visual
    else:
        raise ValueError(f"Unknown ID distribution: {in_dist}")
    return net, transform


def build_hparams_space(num_losses):
    space = {
        f"lossw_{i}": hp.loguniform(f"lossw_{i}", np.log(1e-4), np.log(10.0))
        for i in range(num_losses)
    }
    space["lr"] = hp.loguniform("lr", np.log(1e-3), np.log(1.0))
    space["momentum"] = hp.uniform("momentum", 0.0, 1.0)
    return space


def evaluate_hparams(net, hyperparams, datasets, batch_size, max_iters, repeats=3, full_eval=False):
    all_val_results = []
    for _ in range(repeats):
        initial_net = copy.deepcopy(net)
        current_net = copy.deepcopy(net)
        initial_net.to(device)
        current_net.to(device)
        optimizer = torch.optim.SGD(
            current_net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"]
        )
        loss_weight_hparams = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(NUM_LOSSES)])
        loss_fn = LayerLoss(loss_weight_hparams)
        train_loader = DataLoader(
            datasets["id"], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        finetune(current_net, initial_net, optimizer, loss_fn, train_loader, max_iters)

        val_results = dict()
        if full_eval:
            eval_datasets = ["source", "id_val", "ood", "test1", "test2", "test3", "test4", "test5"]
        else:
            eval_datasets = ["ood_subset_for_hp"]
        for name in eval_datasets:
            if name not in datasets.keys():
                continue
            dataset = datasets[name]
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            loss, accuracy = evaluate(current_net, loader)
            del loader
            val_results[f"{name}_loss"] = loss
            val_results[f"{name}_accuracy"] = accuracy
            all_val_results.append(val_results)
    all_val_results = [{k: np.mean([r[k] for r in all_val_results]) for k in all_val_results[0]}]

    test_accs = []
    if full_eval:
        for name in eval_datasets:
            if name not in datasets.keys():
                continue
            losses = [r[f"{name}_loss"] for r in all_val_results]
            accs = [r[f"{name}_accuracy"] for r in all_val_results]
            if "test" in name:
                test_accs.append(np.mean([r[f"{name}_accuracy"] for r in all_val_results]))
            print(f"{name:10s} loss: {np.mean(losses):.3f} +- {np.std(losses):.3f}  acc: {np.mean(accs):.2f} +- {np.std(accs):.2f}")
        print(f"Average Test Accuracy: {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}")
        print()
    return all_val_results


def run_hyperopt_optimization(net, all_datasets, batch_size, max_iters, max_evals_range, default_hparams, num_losses):
    def hp_objective_fn(hparams):
        _net = copy.deepcopy(net)
        val_results = evaluate_hparams(_net, hparams, all_datasets, batch_size, max_iters)
        val_accs = [r["ood_subset_for_hp_accuracy"] for r in val_results]
        return -np.mean(val_accs)  # maximize accuracy

    space = build_hparams_space(num_losses)
    trials = Trials()

    for max_evals in max_evals_range:
        best_hparams = fmin(
            fn=hp_objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )
        results = evaluate_hparams(net, best_hparams, all_datasets, batch_size, max_iters, repeats=5, full_eval=True)

        for i in range(num_losses):
            best_hparams[f"lossw_{i}"] = np.exp(best_hparams[f"lossw_{i}"])
        best_hparams["lr"] = np.exp(best_hparams["lr"])
        print_hyperparams(best_hparams)

    return results

def print_hyperparams(best_hparams):
    print("\nBest Hyperparameters:")
    for key, value in best_hparams.items():
        print(f"{key}: {value:.4f}")


def main(args):
    net, transform = get_network(in_dist=args.id)
    num_examples_per_dist = {
        "pretrain" : args.num_pretrain_examples,
        "id": args.num_id_examples,
        "id_val": args.num_id_val_examples,
        "id_unlabeled": args.num_id_unlabeled_examples,
        "ood": args.num_ood_examples,
        "ood_unlabeled": args.num_ood_unlabeled_examples,
        "test": args.num_test_examples
    }
    start_time = time.time()
    all_datasets = get_all_datasets(
        root=args.root_dir, pretrain=args.pretrain, id=args.id, id_unlabeled=args.id_unlabeled,
        ood=args.ood, ood_unlabeled=args.ood_unlabeled, test=args.test,
        num_ood_for_hp=args.num_ood_for_hp, num_examples=num_examples_per_dist, transform=transform)
    print(f"Time to load datasets: {time.time() - start_time:.2f} seconds")

    if args.plot:
        num_ood_for_hp_list = [50, 100, 1000, 2000, 4000, 6000, 8000, 10000]
        all_results = {"FT": [], "Ours": []}

        eval_datasets = ["test1", "test2", "test3", "test4", "test5"]
        for num_ood_for_hp in num_ood_for_hp_list:
            all_datasets["ood_subset_for_hp"] = get_subset(all_datasets["ood"], num_ood_for_hp)
            id_and_ood_data = torch.utils.data.ConcatDataset([all_datasets["id"], all_datasets["ood_subset_for_hp"]])
            all_datasets_w_idood = copy.deepcopy(all_datasets)
            all_datasets_w_idood["id"] = id_and_ood_data

            default_hparams = {f"lossw_{i}": 0.0 for i in range(NUM_LOSSES)}
            default_hparams["lossw_0"] = 1.0
            default_hparams["lr"] = args.lr
            default_hparams["momentum"] = 0.9
            print(f"\nID+OOD fine-tune baseline:")
            results = evaluate_hparams(net, default_hparams, all_datasets_w_idood, args.batch_size, args.max_iters, repeats=5, full_eval=True)
            all_results["FT"].append(np.mean([results[-1][f"{eval_dataset}_accuracy"] for eval_dataset in eval_datasets]))

            results = run_hyperopt_optimization(net, all_datasets, args.batch_size, args.max_iters, range(10, 100, 10), default_hparams, NUM_LOSSES)
            all_results["Ours"].append(np.mean([results[-1][f"{eval_dataset}_accuracy"] for eval_dataset in eval_datasets]))

        plot_accuracies(all_results, num_ood_for_hp_list)

        print(all_results)

    else:
        # ID + OOD FT baseline. Default hparams and cross-entropy.
        id_and_ood_data = torch.utils.data.ConcatDataset([all_datasets["id"], all_datasets["ood_subset_for_hp"]])
        all_datasets_w_idood = copy.deepcopy(all_datasets)
        all_datasets_w_idood["id"] = id_and_ood_data
        print(f"\nID+OOD fine-tune baseline:")
        default_hparams = {f"lossw_{i}": 0.0 for i in range(NUM_LOSSES)}
        default_hparams["lossw_0"] = 1.0
        default_hparams["lr"] = args.lr
        default_hparams["momentum"] = 0.9
        evaluate_hparams(net, default_hparams, all_datasets_w_idood, args.batch_size, args.max_iters, repeats=5, full_eval=True)

        print(f"\nID fine-tune baseline:")
        evaluate_hparams(net, default_hparams, all_datasets, args.batch_size, args.max_iters, repeats=5, full_eval=True)

        run_hyperopt_optimization(net, all_datasets, args.batch_size, args.max_iters, range(10, 100, 10), default_hparams, NUM_LOSSES)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/iris/u/cchoi1/Data')
    parser.add_argument('--pretrain', type=str, default='svhn', choices=['svhn', 'clip'])
    parser.add_argument('--num_pretrain_examples', type=int, default=10000)
    parser.add_argument('--id', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--id_unlabeled', nargs='+')
    parser.add_argument('--num_id_examples', type=int, default=-1)
    parser.add_argument('--num_id_val_examples', type=int, default=10000)
    parser.add_argument('--num_id_unlabeled_examples', type=int, default=0)
    parser.add_argument('--ood', nargs='+')
    parser.add_argument('--ood_unlabeled', nargs='+')
    parser.add_argument('--num_ood_examples', type=int, default=10000)
    parser.add_argument('--num_ood_unlabeled_examples', type=int, default=0)
    parser.add_argument('--num_ood_for_hp', type=int, default=100)
    parser.add_argument('--test', nargs='+')
    parser.add_argument('--num_test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    print(args.test)
    # Sample command: python main.py --test mnistc-motion_blur mnistc-impulse_noise mnistc-canny_edges rotated_mnist colored_mnist --plot
    # assert args.id == "mnist" and args.test != ['motion_blur', 'impulse_noise', 'canny_edges', 'rotated_mnist', 'colored_mnist']
    set_seed()
    main(args)
