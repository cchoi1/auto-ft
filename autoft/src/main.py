import sys
sys.path.append("/iris/u/cchoi1/robust-optimizer/autoft")

import copy
from src.models.modeling import ClassificationHead, CLIPEncoder
from src.models.utils import fisher_load
from src.args import parse_arguments
import logging
import random

import os

import src.datasets as datasets
import torch
from src.models.modeling import ImageClassifier
from src.models.utils import cosine_lr, LabelSmoothing
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.finetune import finetune
from src.models.autoft import auto_ft
from losses.layerloss import LayerLoss
from src.models.train_utils import get_subset
import numpy as np

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
    # all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood_subset_for_hp": ood_subset_for_hp}
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
    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    torch.cuda.empty_cache()

    # Datasets
    all_datasets = get_datasets(args, preprocess_fn)

    # Hyperparameters
    hparams = {f"lossw_{i}": 0.0 for i in range(args.num_losses)}
    hparams["lossw_0"] = 1.0
    hparams["lr"] = args.lr
    hparams["momentum"] = 0.9
    loss_weight_hparams = torch.tensor([hparams[f"lossw_{i}"] for i in range(args.num_losses)])
    initial_model = copy.deepcopy(model)
    params = [p for p in model.parameters() if p.requires_grad]

    if args.method == "ft-id":
        # loss_fn = LayerLoss(loss_weight_hparams, initial_model)
        loss_fn = torch.nn.CrossEntropyLoss()
        dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
        # finetune(args, image_classifier, model, loss_fn, optimizer, dataloader, input_key, print_every)
        finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every)
    elif args.method == "ft-id-ood":
        # loss_fn = LayerLoss(loss_weight_hparams, initial_model)
        loss_fn = torch.nn.CrossEntropyLoss()
        id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        dataloader = get_dataloader(id_ood_dataset, is_train=True, args=args, image_encoder=None)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        # finetune(args, image_classifier, model, loss_fn, optimizer, dataloader, input_key, print_every)
        finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every)
    elif args.method == "autoft":
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        ood_hp_dataloader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=True, args=args, image_encoder=None)
        auto_ft(args, model, id_dataloader, ood_hp_dataloader,
                max_evals_range=range(args.val_freq, args.val_freq * args.epochs, args.val_freq), input_key=input_key)
    else:
        raise ValueError("Invalid method")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
