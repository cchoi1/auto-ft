import json
import logging
import os
import time

import numpy as np
import src.datasets as datasets
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, get_autoft_dataloaders
from src.datasets.laion import get_data
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import get_num_classes, test_metric_str, set_seed
from src.models.zeroshot import get_zeroshot_classifier


def initialize_model(args):
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        model = image_classifier.classification_head
        preprocess_fn = image_classifier.val_preprocess
    else:
        model = image_classifier
        preprocess_fn = image_classifier.train_preprocess
        image_classifier.process_images = True
    print('preprocess fn', preprocess_fn)

    devices = list(range(torch.cuda.device_count()))
    print(f"Using devices {devices}.")

    params = [p for p in model.parameters() if p.requires_grad]
    num_parameters = sum([p.numel() for p in params])
    gb_estimate = num_parameters * 4 / (1024 ** 3)
    print(f"Got {args.model} model with {num_parameters:.1e} parameters; {gb_estimate:.3f} GB estimated memory usage")

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model, preprocess_fn


def extract_fewshot_samples(iterator, args):
    images0, labels0, texts0, images1, labels1, texts1 = [], [], [], [], [], []
    match = None

    while True:
        batch = next(iterator)
        image, label, text = batch
        if match is None:
            match = text[0, :]

        for i in range(text.shape[0]):
            if torch.equal(match, text[i]):
                if len(texts0) < args.k:
                    images0.append(image[i])
                    labels0.append(label[i])
                    texts0.append(text[i])
            else:
                if len(texts1) < args.k:
                    images1.append(image[i])
                    labels1.append(label[i])
                    texts1.append(text[i])

        if len(images0) == args.k and len(images1) == args.k:
            break

    return torch.stack(images0 + images1, dim=0), torch.stack(labels0 + labels1, dim=0), torch.stack(texts0 + texts1, dim=0)


def extract_fewshot_samples2(iterator, args):
    class_samples = {}
    num_classes = get_num_classes(args)
    num_classes_collected = 0

    while True:
        batch = next(iterator)
        images, labels = batch

        for i in range(images.shape[0]):
            label = labels[i].item()
            if label not in class_samples.keys():
                class_samples[label] = {"images": [], "labels": []}

            if len(class_samples[label]["images"]) < args.k:
                class_samples[label]["images"].append(images[i])
                class_samples[label]["labels"].append(labels[i])

            # Check if this class just reached args.k samples
            if len(class_samples[label]["images"]) == args.k and len(class_samples[label]["labels"]) == args.k:
                num_classes_collected += 1
                # This class is done, so set its length to args.k + 1 to prevent re-entry into this condition
                class_samples[label]["labels"].append(None)

        if num_classes_collected == num_classes:
            break

    final_images = []
    final_labels = []
    for data in class_samples.values():
        final_images.extend(data["images"][:args.k])
        final_labels.extend(data["labels"][:args.k])

    return torch.stack(final_images, dim=0), torch.stack(final_labels, dim=0)


def get_fewshot_datasets(args, model, preprocess_fn):
    orig_batch_size = args.batch_size
    args.batch_size = args.k
    if args.ft_data is not None:
        train_preprocess_fn = model.module.image_encoder.train_preprocess
        val_preprocess_fn = model.module.image_encoder.val_preprocess
        img_text_data = get_data(args, (train_preprocess_fn, val_preprocess_fn), epoch=0)
        id_dataloader = img_text_data['train_ft'].dataloader
    else:
        id_dataset_class = getattr(datasets, args.id)
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location)
        id_dataloader = get_dataloader(id_dataset, is_train=True, args=args, image_encoder=None)
    id_image, id_label, id_text = extract_fewshot_samples(iter(id_dataloader), args)
    # id_dataset = TensorDataset(id_image, id_label, id_text)

    val_dataset_name = next((dataset_name for dataset_name in args.eval_datasets if 'Val' in dataset_name), None)
    assert val_dataset_name, "Please specify the val dataset in args.eval_datasets."
    val_dataset_class = getattr(datasets, val_dataset_name)
    val_dataset = val_dataset_class(preprocess_fn, location=args.data_location, batch_size=args.k)
    val_dataloader = get_dataloader(val_dataset, is_train=False, args=args, image_encoder=None)
    val_image, val_label = extract_fewshot_samples2(iter(val_dataloader), args)
    # val_dataset = TensorDataset(val_image, val_label)

    ood_dataset_class = getattr(datasets, args.ood)
    ood_dataset = ood_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location, batch_size=args.batch_size)
    ood_dataloader = get_dataloader(ood_dataset, is_train=True, args=args, image_encoder=None)
    ood_image, ood_label = extract_fewshot_samples2(iter(ood_dataloader), args)
    # ood_dataset = TensorDataset(ood_image, ood_label)

    # return {"id": id_dataset, "id_val": val_dataset, "ood_subset_for_hp": ood_dataset}
    args.batch_size = orig_batch_size
    return {"id": (id_image, id_label, id_text), "id_val": (val_image, val_label), "ood_subset_for_hp": (ood_image, ood_label)}


def get_datasets(args, model, preprocess_fn):
    # Few-shot classification
    if args.k is not None:
        all_datasets = get_fewshot_datasets(args, model, preprocess_fn)
        return all_datasets

    # Full setting
    id_dataset_class = getattr(datasets, args.id)
    if args.ft_data is not None:
        train_preprocess_fn = model.module.image_encoder.train_preprocess
        val_preprocess_fn = model.module.image_encoder.val_preprocess
        print('train_preprocess_fn', train_preprocess_fn)
        print('val_preprocess_fn', val_preprocess_fn)
        id_dataset = get_data(args, (train_preprocess_fn, val_preprocess_fn), epoch=0)
    else:
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, n_examples=args.num_id_examples,
                                      location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    if args.unlabeled_id is not None:
        id_unlabeled_dataset_class = getattr(datasets, args.unlabeled_id)
        id_unlabeled_dataset = id_unlabeled_dataset_class(preprocess=preprocess_fn, train=True,
                                                          n_examples=args.num_id_unlabeled_examples,
                                                          location=args.data_location, batch_size=args.batch_size,
                                                          num_workers=args.workers)
    id_val_dataset = id_dataset_class(preprocess=preprocess_fn, train=False, n_examples=args.num_id_examples,
                                      location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    ood_dataset_class = getattr(datasets, args.ood)
    ood_dataset_kwargs = {"preprocess": preprocess_fn, "train": True, "n_examples": args.num_ood_hp_examples,
                          "use_class_balanced": args.use_class_balanced_ood, "location": args.data_location,
                          "batch_size": args.batch_size, "num_workers": args.workers}
    if args.ood == args.id: # Use the test split of the ID dataset as OOD for CIFAR-10
        ood_dataset_kwargs["train"] = False
    else:
        if args.ood == "CIFAR10C":
            ood_dataset_kwargs["severity"] = args.severity
    ood_subset_for_hp = ood_dataset_class(**ood_dataset_kwargs)

    all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    if args.unlabeled_id is not None:
        all_datasets["id_unlabeled"] = id_unlabeled_dataset

    return all_datasets


def train(args, model, preprocess_fn):
    if args.freeze_encoder:
        input_key = 'features'
    else:
        input_key = 'images'
    all_datasets = get_datasets(args, model, preprocess_fn)
    dataset_size_str = ", ".join([f"{k}: {len(all_datasets[k])}" for k in all_datasets.keys()])
    print(f"Got datasets with size {dataset_size_str}")

    params = [p for p in model.parameters() if p.requires_grad]
    image_encoder = None if not args.freeze_encoder else ImageClassifier.load(args.load).image_encoder
    if args.method == "ft-id":
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=image_encoder)
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=image_encoder)
        dataloaders = {"id": id_dataloader, "id_val": id_val_dataloader, "unlabeled": None}
        ft_model, val_metric = finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every=100)
    elif args.method == "ft-id-ood":
        loss_fn = torch.nn.CrossEntropyLoss()
        if hasattr(all_datasets["ood_subset_for_hp"], "dataset"):
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"].dataset])
        else:
            id_ood_dataset = torch.utils.data.ConcatDataset([all_datasets["id"].dataset, all_datasets["ood_subset_for_hp"]])
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        id_dataloader = get_dataloader(id_ood_dataset, is_train=True, args=args, image_encoder=image_encoder)
        id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=image_encoder)
        dataloaders = {"id": id_dataloader, "id_val": id_val_dataloader, "unlabeled": None}
        ft_model, val_metric = finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every=100)
    elif args.method == "autoft":
        fs_id_dataset, fs_val_dataset = None, None
        if args.k is not None:
            fs_id_dataset = all_datasets["id"]
            fs_val_dataset = all_datasets["id_val"]
        dataloaders = get_autoft_dataloaders(args, all_datasets)
        ft_model, val_metric = auto_ft(args, model, dataloaders, all_datasets["ood_subset_for_hp"], args.autoft_epochs, input_key, fs_id_dataset, fs_val_dataset)
    else:
        raise ValueError("Invalid method")
    del all_datasets
    torch.cuda.empty_cache()

    return ft_model, val_metric


def test(model, args):
    model.eval()
    args.current_epoch = args.ft_epochs
    if not args.no_regenerate_head:
        with torch.no_grad():
            classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
            classification_head = classification_head.cuda()
    else:
        classification_head = model.module.classification_head
    eval_results = evaluate(model, classification_head, args)
    os.makedirs(args.save, exist_ok=True)
    results_path = os.path.join(args.save, "eval_results.json")
    with open(results_path, 'w') as f:
        f.write(json.dumps(eval_results))
    print(f'\nSaved evaluation results to {results_path}.')
    return eval_results[test_metric_str(args)]


def main(args):
    assert "IDVal" not in args.eval_datasets, "IDVal must be specified as an evaluation dataset"
    logger = logging.getLogger('main')
    logger = setup_logging(args, logger)
    args_dict = dict(sorted(vars(args).items()))
    args_str = "\n".join([f"{k:30s}: {v}" for k, v in args_dict.items()])
    logger.info(f"args:\n{args_str}")

    test_metrics = []
    for i in range(args.repeats):
        set_seed(args.seed + i)
        print(f"\nRun {i + 1} / {args.repeats}")
        model, preprocess_fn = initialize_model(args)
        if not args.eval_only:
            model, val_metric = train(args, model, preprocess_fn)
        test_metric = test(model, args)
        test_metrics.append(test_metric)
        print(f"Run {i + 1}. \n\tVal metric: {val_metric} \n\tTest metric: {test_metric}"
              f"\n\tTest metrics thus far: {test_metrics}")

    print(f"Test metrics: {test_metrics}")
    avg_test_metric, std_test_metric = np.mean(test_metrics), np.std(test_metrics)
    print(f"Average test metric: {avg_test_metric:.3f} +/- {std_test_metric:.3f}")
    logger.info(f"Average test metric: {avg_test_metric:.3f} +/- {std_test_metric:.3f}")


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()
    main(args)
    print(f"\nRUN TIME: {time.time() - start_time:.3f}")
