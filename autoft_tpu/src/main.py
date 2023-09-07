import time
import json
import os

import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.utils import get_ood_datasets
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import is_tpu_available, get_device, save_model
from src.models.sampler import get_sampler

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

def configure_device(model, rank=0):
    if is_tpu_available():
        device = xm.xla_device(rank)
    else:
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)
        xm.master_print('Using devices', devices)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else 'cpu')
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

    params = [p for p in model.parameters() if p.requires_grad]
    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        ft_model, ft_eval_results = finetune_final(args, model, loss_fn, optimizer, all_datasets["id"], input_key, print_every)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        ft_model, ft_eval_results = finetune_final(args, model, loss_fn, optimizer, id_ood_dataset, input_key, print_every)
    elif args.method == "autoft":
        # id_sampler = get_sampler(all_datasets["id"], None, shuffle=True)
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        # ood_hp_sampler = get_sampler(all_datasets["ood_subset_for_hp"], None, shuffle=True)
        ood_hp_dataloader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=True, args=args, image_encoder=None)
        ft_model, ft_eval_results = auto_ft(args, model, id_dataloader, ood_hp_dataloader, max_evals=args.autoft_epochs, input_key=input_key, print_every=100)
    else:
        raise ValueError("Invalid method")

    return ft_model, ft_eval_results

def test(args, model, all_eval_results, total_steps, logger):
    """Evaluate the model on the test set and save the results."""
    should_eval = xm.is_master_ordinal() if is_tpu_available() else True
    if should_eval:
        args.current_epoch = args.ft_epochs
        eval_results = evaluate(model.module, args)
        xm.master_print(eval_results)
        logger.info(json.dumps(eval_results, indent=4))
        all_eval_results[total_steps] = eval_results
        os.makedirs(args.save, exist_ok=True)
        results_path = os.path.join(args.save, 'eval_results.json')
        with open(results_path, 'wb') as f:
            f.write(json.dumps(all_eval_results))
        xm.master_print(f'\nSaved evaluation results to {results_path}.')

def main(args, rank=0):
    logger = setup_logging(args)
    logger.info(args)
    model, preprocess_fn = initialize_model(args)
    model, device = configure_device(model, rank)
    if args.eval_only:
        model.load(args.load)
        return test(args, model, {}, 0, logger)
    ft_model, ft_eval_results = train(args, model, preprocess_fn)
    save_model(args, model, logger)
    return test(args, ft_model, ft_eval_results, args.ft_epochs, logger)


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    # if is_tpu_available():
    #     print("TPU AVAILABLE")
    #     xmp.spawn(main, args=(args,), nprocs=8, start_method='spawn')
    # else:
    main(args)

    print(f"\nRUN TIME: {time.time() - start_time:.3f}", flush=True)
