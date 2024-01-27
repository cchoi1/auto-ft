import os
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Method Selection
    parser.add_argument("--method", type=str, choices=["autoft", "ft-id", "ft-id-ood", "zeroshot", "flyp"])

    # Dataset Configuration
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'), help="Root directory for datasets.")
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--id_val", type=str, default=None)
    parser.add_argument("--num_id_examples", type=int, default=-1)
    parser.add_argument("--num_id_val_examples", type=int, default=-1)
    parser.add_argument("--ood", type=str, default=None)
    parser.add_argument("--num_ood_hp_examples", type=int, default=-1)
    parser.add_argument("--eval-datasets", type=lambda x: x.split(","), default=None, help="Comma-separated datasets for evaluation.")
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "auto"], default="auto", help="Type of dataset to process.")
    parser.add_argument("--severity", type=int, default=5, help="Severity level for CIFAR10C corruption.")

    # Training Configuration
    parser.add_argument("--model", type=str, default=None, help="Model type (e.g., RN50, ViT-B/32).")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--losses", nargs="+", choices=["ce", "hinge", "entropy", "dcm", "flyp", "l1zero", "l2zero", "l1init", "l2init"])
    parser.add_argument("--layerwise_loss", action="store_true")
    parser.add_argument("--layerwise_opt", action="store_true")
    parser.add_argument("--load_hparams", type=str, help="Path to hyperparameters.")
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--hopt_evals", type=int, default=10)
    parser.add_argument("--inner_steps", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting.")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--use_class_balanced_ood", action="store_true")
    parser.add_argument("--optuna_sampler", type=str, default="TPESampler")
    parser.add_argument("--clip_gradient", action="store_true")
    parser.add_argument("--no_lr_wd", action="store_true")
    parser.add_argument("--autoft_repeats", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    # Saving and Logging
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--load", type=lambda x: x.split(","), default=None, help="Load classifiers, separated by commas.")
    parser.add_argument("--save", type=str, default="./saved", help="Directory for saving models and results.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs.")

    # Additional Configuration
    parser.add_argument("--ft_data", type=str, default=None, help="Path to CSV file with training data.")
    parser.add_argument("--val_data", type=str, default=None, help="Path to CSV file with validation data.")
    parser.add_argument("--template", type=str, default=None, help="Prompt template for zero-shot learning.")
    parser.add_argument("--classnames", type=str, default="openai", help="Class names to use.")
    parser.add_argument("--alpha", nargs='*', type=float, default=[0.5], help="Interpolation coefficients for ensembling.")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing value.")
    parser.add_argument("--warmup_length", type=int, default=500)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the image encoder during fine-tuning.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for features and encoder.")
    parser.add_argument("--fisher", type=lambda x: x.split(","), default=None, help="Fisher information options.")
    parser.add_argument("--fisher_floor", type=float, default=1e-8, help="Floor value for Fisher information.")
    parser.add_argument("--ce_ablation", action="store_true")
    parser.add_argument("--train-num-samples", type=int, default=None, help="Number of training samples for webdataset.")
    parser.add_argument("--k", type=int, default=None, help="k-value for few-shot ImageNet.")
    parser.add_argument("--csv-separator", type=str, default="\t", help="Separator for CSV datasets.")
    parser.add_argument("--csv-img-key", type=str, default="filepath", help="Key for image paths in CSV datasets.")
    parser.add_argument("--csv-caption-key", type=str, default="title", help="Key for captions in CSV datasets.")
    parser.add_argument("--clip_load", type=str, default=None, help="Load finetuned CLIP model.")
    parser.add_argument("--wise_save", type=str, default=None, help="Save path for wiseft results.")
    parser.add_argument("--get_labeled_csv", action="store_true", help="Extract labels from CSV.")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum learning rate for cosine scheduler.")

    # Parse arguments
    parsed_args = parser.parse_args()

    # Post-processing steps
    parsed_args.losses = sorted(parsed_args.losses)
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
