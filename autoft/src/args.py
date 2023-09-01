import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["autoft", "ft-id", "ft-id-ood", "zeroshot"])

    # Datasets
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'),
                        help="The root directory for the datasets.")
    parser.add_argument("--id", default=None, type=str)
    parser.add_argument("--num_id_examples", default=-1, type=int)
    parser.add_argument("--num_id_val_examples", default=-1, type=int)
    parser.add_argument("--ood", default=None, type=str)
    parser.add_argument("--num_ood_examples", default=-1, type=int)
    parser.add_argument("--num_ood_hp_examples", default=-1, type=int)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same model used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "auto"], default="auto",
                        help="Which type of dataset to process.")
    parser.add_argument("--severity", type=int, default=5, help="Severity of corruption for CIFAR10C.")

    # Training
    parser.add_argument("--model", type=str, default=None, help="The type of model (e.g. RN50, ViT-B/32).")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--num_losses", type=int, default=8)
    parser.add_argument("--loss_type", type=str, choices=["LearnedLoss", "LayerwiseLoss"], default="LearnedLoss")
    parser.add_argument("--pointwise_loss", action="store_true")
    parser.add_argument("--load_hparams", type=str, help="Path to hyperparameters to load.")
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--autoft_epochs", type=int, default=10)
    parser.add_argument("--inner_steps", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training.")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Plot results.")

    # Saving/Logging
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help=
        "Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./saved",
        help=
        "Directory to save models and results. If not provided, will not save anything.",
    )

    # Reproduction
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument("--run", type=int, default=1, help="Repeated run number")

    # TODO other args from FLYP code that we don't need right now but might need later
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=
        "Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=
        ('Interpolation coefficient for ensembling. '
         'Users should specify N-1 values, where N is the number of '
         'models being ensembled. The specified numbers should sum to '
         'less than 1. Note that the order of these values matter, and '
         'should be the same as the order of the classifiers being ensembled.'
         ))
    parser.add_argument("--ls",
                        type=float,
                        default=0.0,
                        help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help=
        "Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )
    parser.add_argument('--ce_ablation', action="store_true")
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )

    parser.add_argument("--k",
                        type=int,
                        default=None,
                        help="k for few shot ImageNet")

    parser.add_argument("--csv-separator",
                        type=str,
                        default="\t",
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.")
    parser.add_argument(
        "--clip_load",
        type=str,
        default=None,
        help="Load finetuned clip",
    )
    parser.add_argument(
        "--wise_save",
        type=str,
        default=None,
        help="Save path for wiseft results",
    )
    parser.add_argument("--get_labeled_csv",
                        default=False,
                        action="store_true",
                        help="get labels from csv.")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
