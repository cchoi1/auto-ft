from src.args import parse_arguments
import random

import os

import src.datasets as datasets
import torch
from src.models.modeling import ImageClassifier
from src.datasets.common import get_dataloader
from src.models.finetune import finetune_final
from src.models.autoft import auto_ft
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

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
    ood_dataset = ood_dataset_class(
        preprocess_fn,
        train=True,
        n_examples=args.num_ood_examples,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    ood_subset_for_hp = ood_dataset_class(
        preprocess_fn,
        train=True,
        n_examples=args.num_ood_hp_examples,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood": ood_dataset, "ood_subset_for_hp": ood_subset_for_hp}
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
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl')

    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert args.save is not None, 'Please provide a path to store models'
    #############################################################################

    # # Initialize the CLIP encoder
    # clip_encoder = CLIPEncoder(args, keep_lang=True)
    # classification_head = ClassificationHead(normalize=True, weights=None)
    logger.info(args)

    # Model
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
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

    # Datasets
    all_datasets = get_datasets(args, preprocess_fn)

    # Hyperparameters
    hparams = {f"lossw_{i}": 0.0 for i in range(args.num_losses)}
    hparams["lossw_0"] = 1.0
    hparams["lr"] = args.lr
    hparams["momentum"] = 0.9
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
        auto_ft(args, model, id_dataloader, ood_hp_dataloader,
                max_evals_range=range(args.val_freq, args.val_freq * args.autoft_epochs, args.val_freq), input_key=input_key)
    else:
        raise ValueError("Invalid method")


if __name__ == '__main__':
    args = parse_arguments()
    set_seed()
    main(args)
