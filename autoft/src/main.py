import json
import logging
import os
import time

import torch

import src.datasets as datasets
from src.args import parse_arguments
from src.datasets.common import get_dataloader, get_autoft_dataloaders
from src.datasets.laion import get_data
from src.datasets.utils import get_ood_datasets
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import extract_from_data_parallel
from torch.utils.data import TensorDataset


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
    print(f"Using devices {devices}.")
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model, preprocess_fn


def extract_fewshot_samples(iterator, args):
    images0, labels0, texts0, images1, labels1, texts1 = [], [], [], [], [], []
    match = None

    while True:
        batch = next(iterator)
        use_text = len(batch) == 3

        image, label = batch[:2]
        text = batch[2] if use_text else [None] * len(label)  # Defaulting to list of Nones if no text

        if match is None and use_text:
            match = text[0]

        for i in range(image.shape[0]):
            is_match_condition = (not use_text and i%2 == 0) or (use_text and torch.equal(match, text[i]))
            if is_match_condition:
                if len(images0) < args.k:
                    images0.append(image[i])
                    labels0.append(label[i])
                    if use_text:
                        texts0.append(text[i])
            else:
                if len(images1) < args.k:
                    images1.append(image[i])
                    labels1.append(label[i])
                    if use_text:
                        texts1.append(text[i])

        if len(images0) == args.k and len(images1) == args.k:
            break

    if not use_text:
        return torch.stack(images0 + images1, dim=0), torch.stack(labels0 + labels1, dim=0)
    return torch.stack(images0 + images1, dim=0), torch.stack(labels0 + labels1, dim=0), torch.stack(texts0 + texts1, dim=0)

def get_fewshot_datasets(args, model, preprocess_fn):
    if args.ft_data is not None:
        temp_model = extract_from_data_parallel(model)
        args.batch_size = 2 * args.k
        img_text_data = get_data(args,
                                 (temp_model.image_encoder.train_preprocess, temp_model.image_encoder.val_preprocess),
                                 epoch=0)
        id_dataloader = img_text_data['train_ft'].dataloader
        del temp_model
        torch.cuda.empty_cache()
    else:
        id_dataset_class = getattr(datasets, args.id)
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location,
                                      batch_size=args.k)
        id_dataloader = get_dataloader(id_dataset, is_train=True, args=args, image_encoder=None)
    id_iterator = iter(id_dataloader)
    id_image, id_label, id_text = extract_fewshot_samples(id_iterator, args)
    print('id few shot shapes', id_image.shape, id_label.shape, id_text.shape)
    id_dataset = TensorDataset(id_image, id_label, id_text)

    val_dataset_name = next((dataset_name for dataset_name in args.eval_datasets if 'Val' in dataset_name), None)
    assert val_dataset_name, "Please specify the val dataset in args.eval_datasets."
    val_dataset_class = getattr(datasets, val_dataset_name)
    val_dataset = val_dataset_class(preprocess_fn, location=args.data_location, batch_size=args.k)
    val_dataloader = get_dataloader(val_dataset, is_train=False, args=args, image_encoder=None)
    val_iterator = iter(val_dataloader)
    val_image, val_text = extract_fewshot_samples(val_iterator, args)
    print('val few shot shapes', val_image.shape, val_text.shape)
    val_dataset = TensorDataset(val_image, val_text)

    ood_dataset_class = getattr(datasets, args.ood)
    ood_dataset = ood_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location, batch_size=args.batch_size)
    ood_dataloader = get_dataloader(ood_dataset, is_train=True, args=args, image_encoder=None)
    # ood_iterator = iter(ood_dataloader)
    # ood_image, ood_text = extract_fewshot_samples(ood_iterator, args)
    # ood_dataset = TensorDataset(ood_image, ood_text)

    return {"id": id_dataset, "id_val": val_dataset, "ood_subset_for_hp": ood_dataset}


def get_datasets(args, model, preprocess_fn):
    if args.k is not None:
        all_datasets = get_fewshot_datasets(args, model, preprocess_fn)
        return all_datasets

    id_dataset_class = getattr(datasets, args.id)
    if args.ft_data is not None:
        temp_model = extract_from_data_parallel(model)
        id_dataset = get_data(args, (temp_model.image_encoder.train_preprocess, temp_model.image_encoder.val_preprocess),
                              epoch=0)
        del temp_model
        torch.cuda.empty_cache()
    else:
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

    all_datasets = get_datasets(args, model, preprocess_fn)
    dataset_size_str = ", ".join([f"{k}: {len(all_datasets[k])}" for k in all_datasets])
    print(f"Got datasets with size {dataset_size_str}")

    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=None)
        dataloaders = {"id": id_dataloader, "id_val": id_val_dataloader}
        model = finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        id_dataloader = get_dataloader(id_ood_dataset, is_train=True, args=args, image_encoder=None)
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=None)
        dataloaders = {"id": id_dataloader, "id_val": id_val_dataloader}
        model = finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every)
    elif args.method == "autoft":
        if args.k is not None:
            args.batch_size = 2 * args.k
        dataloaders = get_autoft_dataloaders(args, all_datasets)
        model = auto_ft(args, model, dataloaders, all_datasets["ood_subset_for_hp"], args.autoft_epochs, input_key)
    else:
        raise ValueError("Invalid method")
    del all_datasets
    torch.cuda.empty_cache()

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
    assert "IDVal" not in args.eval_datasets, "IDVal must be specified as an evaluation dataset"
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
