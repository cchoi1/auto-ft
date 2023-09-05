import logging
import os
import time

import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.utils import get_ood_datasets
from src.models.autoft import auto_ft
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import is_tpu_available

devices = list(range(torch.cuda.device_count()))
device = xm.xla_device()

def setup_logging(args):
    save_dir = os.path.join(args.save, args.id, args.method)
    os.makedirs(save_dir, exist_ok=True)
    if args.method == "autoft":
        method_name = f"ood{args.ood}_{args.loss_type}"
        if args.pointwise_loss:
            method_name += "_pw"
        if args.num_ood_unlabeled_examples is not None:
            method_name += "_unlabeled"
        run_details = f"no{args.num_ood_hp_examples}_nou{args.num_ood_unlabeled_examples}_afep{args.autoft_epochs}_is{args.inner_steps}_ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}"
        args.save = os.path.join(save_dir, method_name, run_details)
    elif args.method == "ft-id-ood":
        method_name = f"ood{args.ood}"
        if args.num_ood_unlabeled_examples is not None:
            method_name += "_unlabeled"
        run_details = f"no{args.num_ood_hp_examples}_nou{args.num_ood_unlabeled_examples}_ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}"
        args.save = os.path.join(save_dir, method_name, run_details)
    elif args.method == "ft-id":
        run_details = f"ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}"
        args.save = os.path.join(save_dir, run_details)
    logging_path = os.path.join("logs", args.save)
    xm.master_print(f"\nMODEL SAVE PATH: {args.save}")
    xm.master_print(f"\nLOGGING PATH: {logging_path}\n")
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def initialize_model(args):
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        model = image_classifier.classification_head
        preprocess_fn = image_classifier.val_preprocess
    else:
        model = image_classifier
        preprocess_fn = image_classifier.train_preprocess
        image_classifier.process_images = True
    return model, preprocess_fn

def configure_device(model):
    if is_tpu_available():
        device = xm.xla_device()
    else:
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)
        xm.master_print('Using devices', devices)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device

def get_datasets(args, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    id_dataset = id_dataset_class(
        preprocess_fn,
        train=True,
        n_examples=args.num_id_examples,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    id_val_dataset = id_dataset_class(
        preprocess_fn,
        train=False,
        n_examples=args.num_id_val_examples,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    ood_dataset_class = getattr(datasets, args.ood)
    n_examples = args.num_ood_hp_examples + args.num_ood_unlabeled_examples if args.num_ood_unlabeled_examples is not None else args.num_ood_hp_examples
    ood_dataset_kwargs = {"preprocess": preprocess_fn, "train": True, "n_examples": n_examples,
                          "location": args.data_location, "batch_size": args.batch_size, "num_workers": args.workers}
    if args.ood == args.id:
        ood_subset_for_hp = id_val_dataset
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

    all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    for i, dataset_name in enumerate(args.eval_datasets):
        test_dataset_class = getattr(datasets, dataset_name)
        all_datasets[f"test{i}"] = test_dataset_class(
            preprocess_fn,
            train=False,
            n_examples=-1,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    return all_datasets


def run_method(args, model, preprocess_fn):
    if args.freeze_encoder:
        input_key = 'features'
        print_every = 1000
    else:
        input_key = 'images'
        print_every = 100

    all_datasets = get_datasets(args, preprocess_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        finetune_final(args, model, loss_fn, optimizer, all_datasets["id"], input_key, print_every)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        finetune_final(args, model, loss_fn, optimizer, id_ood_dataset, input_key, print_every)
    elif args.method == "autoft":
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        ood_hp_dataloader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=True, args=args, image_encoder=None)
        if args.plot:
            auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals=args.autoft_epochs, input_key=input_key, print_every=100)
        else:
            auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals=args.autoft_epochs, input_key=input_key)
    else:
        raise ValueError("Invalid method")

def main(args, rank=None):
    logger = setup_logging(args)
    logger.info(args)
    model, preprocess_fn = initialize_model(args)
    model, device = configure_device(model)
    run_method(args, model, preprocess_fn)


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    if is_tpu_available():
        print("TPU AVAILABLE")
        xmp.spawn(main, args=(args,), nprocs=8, start_method='fork')
    else:
        main(args)

    print(f"\nRUN TIME: {time.time() - start_time:.3f}", flush=True)
