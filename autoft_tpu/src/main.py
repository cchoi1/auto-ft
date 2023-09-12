import os
import re
import time

import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
from src.args import parse_arguments
from src.datasets.utils import get_ood_datasets
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.utils import initialize_model


def get_datasets(args, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    id_dataset = id_dataset_class(preprocess_fn, train=True, n_examples=args.num_id_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    ood_dataset_class = getattr(datasets, args.ood)
    n_examples = -1 if args.num_ood_unlabeled_examples is not None else args.num_ood_hp_examples
    ood_dataset_kwargs = {"preprocess": preprocess_fn, "train": True, "n_examples": n_examples,
                          "location": args.data_location, "batch_size": args.batch_size, "num_workers": args.workers}
    if args.ood == args.id: # Use the test split of the ID distribution as OOD
        ood_dataset_kwargs["train"] = False
    else:
        if args.ood == "CIFAR10C":
            ood_dataset_kwargs["severity"] = args.severity
    ood_dataset = ood_dataset_class(**ood_dataset_kwargs)
    if args.num_ood_unlabeled_examples is not None:
        ood_labeled_dataset, ood_unlabeled_dataset = get_ood_datasets(
            ood_dataset.dataset,
            args.num_ood_hp_examples,
            args.num_ood_unlabeled_examples
        )
        ood_subset_for_hp = torch.utils.data.ConcatDataset([ood_labeled_dataset, ood_unlabeled_dataset])
    else:
        ood_subset_for_hp = ood_dataset

    all_datasets = {"id": id_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    return all_datasets


def train(args, model, preprocess_fn):
    if args.freeze_encoder:
        input_key = 'features'
        print_every = 1000
    else:
        input_key = 'images'
        print_every = 100

    all_datasets = get_datasets(args, preprocess_fn)
    print("Fetched all datasets")

    ft_eval_results = {}
    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        ft_eval_results = finetune(args, model, loss_fn, optimizer, all_datasets["id"], input_key)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        ft_eval_results = finetune(args, model, loss_fn, optimizer, id_ood_dataset, input_key)
    elif args.method == "autoft":
        ft_eval_results = auto_ft(args, model, all_datasets["id"], all_datasets["ood_subset_for_hp"], max_evals=args.autoft_epochs, input_key=input_key)
    else:
        raise ValueError("Invalid method")

    return ft_eval_results


def test(model, ckpt_path, logger):
    model.load(ckpt_path)
    xm.master_print(f"Evaluating checkpoint at {ckpt_path}...")
    logger.info(f"Evaluating checkpoint at {ckpt_path}...")
    evaluate(model, args, spawn_required=True)


def main(args, rank=0):
    logger = setup_logging(args)
    logger.info(args)
    model, preprocess_fn = initialize_model(args, rank)
    if args.eval_only:
        return test(args, args.load, logger)
    else:
        ft_eval_results = train(args, model, preprocess_fn)
        ckpts = [f for f in os.listdir(args.save) if re.match(r'checkpoint_\d+\.pt', f)]
        if ckpts:
            last_ckpt = max(ckpts, key=lambda x: int(re.search(r'(\d+)', x).group()))
            ckpt_path = os.path.join(args.save, last_ckpt)
        return test(model, ckpt_path, logger)


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    os.environ["PYTHONPATH"] = "${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft_tpu/"
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    main(args)
    xm.master_print("Run time: {:.3f} s".format(time.time() - start_time))