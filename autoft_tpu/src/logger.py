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
from src.models.utils import is_tpu_available, get_device
from src.models.sampler import get_sampler

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