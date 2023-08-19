import argparse
from typing import List
from data.mnist_c import _CORRUPTIONS

DISTS = ["mnist", "mnistc", "mnist-label-shift", "emnist", "svhn", "svhn-grayscale", "colored_mnist", "rotated_mnist"] \
        + _CORRUPTIONS

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs='+', type=str,
                        help="Choose a subset (maximum size=2) of [full, surgical, ours, ours-avg, pretrained, lp-ft, wise-ft].")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--pretrain", type=str, choices=DISTS)
    parser.add_argument("--id", type=str, choices=DISTS)
    parser.add_argument("--ood", type=str, choices=DISTS)
    parser.add_argument("--test", type=str, choices=DISTS)
    parser.add_argument("--src_samples_per_class", type=int, default=-1, help="Number of SRC samples per class, -1 for all.")
    parser.add_argument("--id_samples_per_class", type=int, default=-1, help="Number of ID samples per class, -1 for all.")
    parser.add_argument("--ood_samples_per_class", type=int, default=-1, help="Number of OOD samples per class, -1 for all.")
    parser.add_argument("--output_channels", type=int)
    parser.add_argument("--optimizer_name", type=str, default=None)
    parser.add_argument("--loss_name", type=str, default=None)
    parser.add_argument("--wnb", action="store_true", help="Learn both weights and biases to scale LRs")
    parser.add_argument("--momentum", action="store_true", help="Learn momentum")
    parser.add_argument("--output", type=str, choices=["lr_multiplier", "update"], default="lr_multiplier")
    parser.add_argument("--ft_dists", nargs='+', type=str, choices=["id", "ood", "id+ood", "src+id", "src+ood", "src+id+ood"])
    parser.add_argument("--features", nargs='+', type=str,
                        help="Choose a subset of [p, g, p_norm, g_norm, depth, wb, dist_init_param, iter, loss, "
                             "loss_ema, tensor_rank, pos_enc, pos_enc_cont, momentum, layer_type].",
                        default=None)
    parser.add_argument("--lopt_net_dim", type=int)
    parser.add_argument("--use_hyperopt", action="store_true")
    parser.add_argument("--meta_steps", type=int, default=100)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--inner_steps_range", type=int, default=None)
    parser.add_argument("--meta_loss_avg_w", type=float, default=0.0)
    parser.add_argument("--meta_loss_final_w", type=float, default=1.0)
    parser.add_argument("--meta_batch_size", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--ft_val_freq", type=int, default=100)
    parser.add_argument("--val_meta_batch_size", type=int, default=20)
    parser.add_argument("--val_inner_steps", type=int, default=20)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--meta_lr", type=float, default=3e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-1)
    parser.add_argument("--ft_lr", type=float)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_nets", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--l2_lambda", type=float, default=None)
    parser.add_argument("--val", nargs='+', type=str, choices=["id", "ood"])
    parser.add_argument("--alpha", type=float, default=0.99, help="Decay factor for the moving average for WiSE-FT.")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="cchoi")
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