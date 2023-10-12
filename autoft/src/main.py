import json
import logging
import os
import time

import torch

import src.datasets as datasets
from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.laion import get_data
from src.datasets.utils import get_ood_datasets
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel


def initialize_model(args):
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        model = image_classifier.classification_head
        preprocess_fn = image_classifier.val_preprocess
    else:
        model = image_classifier
        preprocess_fn = image_classifier.train_preprocess
        image_classifier.process_images = True
    devices = list(range(torch.cuda.device_count()))
    print(f"devices {devices}")
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model, preprocess_fn

def get_datasets(args, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, n_examples=args.num_id_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    if args.unlabeled_id is not None:
        id_unlabeled_dataset_class = getattr(datasets, args.unlabeled_id)
        id_unlabeled_dataset = id_unlabeled_dataset_class(preprocess=preprocess_fn, train=True, n_examples=args.num_id_unlabeled_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    id_val_dataset = id_dataset_class(preprocess=preprocess_fn, train=False, n_examples=args.num_id_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    ood_dataset_class = getattr(datasets, args.ood)
    n_examples = -1 if args.num_ood_unlabeled_examples is not None else args.num_ood_hp_examples
    ood_dataset_kwargs = {"preprocess": preprocess_fn, "train": True, "n_examples": n_examples,
                          "use_class_balanced": args.use_class_balanced_ood, "location": args.data_location,
                          "batch_size": args.batch_size, "num_workers": args.workers}
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

    all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    if args.unlabeled_id is not None:
        all_datasets["id_unlabeled"] = id_unlabeled_dataset

    return all_datasets


def train(args, model, preprocess_fn):
    if args.freeze_encoder:
        input_key = 'features'
        print_every = 1000
    else:
        input_key = 'images'
        print_every = 100

    params = [p for p in model.parameters() if p.requires_grad]
    num_parameters = sum([p.numel() for p in params])
    gb_estimate = num_parameters * 4 / (1024 ** 3)
    print(f"Got {args.model} model with {num_parameters:.1e} parameters; {gb_estimate:.3f} GB estimated memory usage")

    all_datasets = get_datasets(args, preprocess_fn)
    dataset_size_str = ", ".join([f"{k}: {len(all_datasets[k])}" for k in all_datasets])
    print(f"Got datasets with size {dataset_size_str}")

    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        model = finetune_final(args, model, loss_fn, optimizer, dataloader, input_key, print_every)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        dataloader = get_dataloader(id_ood_dataset, is_train=True, args=args, image_encoder=None)
        model = finetune_final(args, model, loss_fn, optimizer, dataloader, input_key, print_every)
    elif args.method == "autoft":
        if args.ft_data is not None:
            temp_model = extract_from_data_parallel(model)
            img_text_data = get_data(args, (temp_model.image_encoder.train_preprocess, temp_model.image_encoder.val_preprocess), epoch=0)
            id_dataloader = img_text_data['train_ft'].dataloader
            del temp_model; torch.cuda.empty_cache()
        else:
            id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        ood_hp_dataloader = get_dataloader(all_datasets["ood_subset_for_hp"], is_train=True, args=args, image_encoder=None)
        if args.unlabeled_id is not None:
            unlabeled_dataloader = all_datasets["id_unlabeled"].dataloader
        else:
            unlabeled_dataloader = None
        model = auto_ft(args, model, id_dataloader, ood_hp_dataloader, all_datasets["ood_subset_for_hp"], args.autoft_epochs, input_key, unlabeled_dataloader)
        del id_dataloader, ood_hp_dataloader, unlabeled_dataloader
    else:
        raise ValueError("Invalid method")
    del all_datasets; torch.cuda.empty_cache()

    return model


def test_finetuned_model(args, logger, model, all_eval_results, total_steps):
    """Evaluate the model on the test set and save the results."""
    args.current_epoch = args.ft_epochs
    eval_results = evaluate(model.module, args)
    print(eval_results)
    logger.info(json.dumps(eval_results, indent=4))
    all_eval_results[total_steps] = eval_results
    os.makedirs(args.save, exist_ok=True)
    results_path = os.path.join(args.save, 'eval_results.json')
    with open(results_path, 'w') as f:
        f.write(json.dumps(all_eval_results))
    print(f'\nSaved evaluation results to {results_path}.')


def main(args):
    logger = logging.getLogger('main')
    logger = setup_logging(args, logger)
    args_dict = dict(sorted(vars(args).items()))
    args_str = "\n".join([f"{k:30s}: {v}" for k, v in args_dict.items()])
    logger.info(f"args:\n{args_str}")
    model, preprocess_fn = initialize_model(args)
    if not args.eval_only:
        model = train(args, model, preprocess_fn)
    test_finetuned_model(args, logger, model, {}, args.ft_epochs)


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    main(args)

    print(f"\nRUN TIME: {time.time() - start_time:.3f}")
