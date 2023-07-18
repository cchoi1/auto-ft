import argparse
from datasets import _CORRUPTIONS

def get_args(): 
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
        default="mnist",
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
        default="LayerSGDLinear",
    )
    parser.add_argument("--ft_id_ood", action="store_true", help="Fine-tune w/ meta-params on both ID and OOD data.")

    parser.add_argument("--features", nargs='+', type=str,
                        help="Choose a subset of [p, g, p_norm, g_norm, depth, wb, dist_init_param, iter, loss, loss_ema, tensor_rank].",
                        default=None)
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_nets", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--l2_lambda", type=float, default=None)
    parser.add_argument("--val", type=str, choices=["id", "ood"])
    parser.add_argument("--no_wandb", action="store_true")
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
    print("\n\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:20s} =  {v}")

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
    return args