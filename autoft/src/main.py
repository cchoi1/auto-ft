import logging
import os
import random
import time

import numpy as np
import src.datasets as datasets
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.utils import get_ood_datasets
from src.models.autoft import auto_ft
from src.models.finetune import finetune_final
from src.models.flyp_loss import flyp_loss
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from torch.nn.parallel import DistributedDataParallel as DDP

devices = list(range(torch.cuda.device_count()))

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


def main(args):
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')

    ###logging##################################################################
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
    print(f"\nMODEL SAVE PATH: {args.save}")
    print(f"\nLOGGING PATH: {logging_path}\n")
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #############################################################################
    logger.info(args)

    if args.method == "flyp":
        clip_encoder = CLIPEncoder(args, keep_lang=True)
        classification_head = ClassificationHead(normalize=True, weights=None)
        return flyp_loss(args, clip_encoder, classification_head, logger)

    # Model
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_classifier.process_images = True
        print_every = 100
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        print('Using device', local_rank)
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices)
        print('Using devices', devices)

    torch.cuda.empty_cache()

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


if __name__ == '__main__':
    args = parse_arguments()
    set_seed()
    start_time = time.time()
    main(args)
    print(f"Total run time: {time.time() - start_time:.3f}", flush=True)
