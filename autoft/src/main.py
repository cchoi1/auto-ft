import json
import logging
import os

import numpy as np
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, get_autoft_dataloaders
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.utils import get_num_classes, test_metric_str, set_seed
from src.models.zeroshot import get_zeroshot_classifier
from src.get_data import get_datasets


def initialize_model(args):
    image_classifier = ImageClassifier.load(args.load)
    preprocess_fn = image_classifier.val_preprocess if args.freeze_encoder else image_classifier.train_preprocess
    model = image_classifier.classification_head if args.freeze_encoder else image_classifier
    if not args.freeze_encoder:
        image_classifier.process_images = True

    devices = list(range(torch.cuda.device_count()))
    print(f"Using devices {devices}.")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gb_estimate = num_parameters * 4 / (1024 ** 3)
    print(f"Got {args.model} model with {num_parameters:.1e} parameters; {gb_estimate:.3f} GB estimated memory usage")

    model = torch.nn.DataParallel(model.cuda(), device_ids=devices)
    return model, preprocess_fn


def print_dataset_sizes(all_datasets):
    dataset_size_str = ", ".join([f"{k}: {len(all_datasets[k])}" for k in all_datasets.keys()])
    print(f"Got datasets with size {dataset_size_str}")


def get_training_components(params, args):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    return loss_fn, optimizer


def prepare_dataloaders(all_datasets, args, image_encoder):
    id_dataloader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=image_encoder)
    id_val_dataloader = get_dataloader(all_datasets["id_val"], is_train=False, args=args, image_encoder=image_encoder)
    return {"id": id_dataloader, "id_val": id_val_dataloader, "unlabeled": None}


def train(args, model, preprocess_fn):
    input_key = 'features' if args.freeze_encoder else 'images'
    all_datasets = get_datasets(args, model, preprocess_fn)
    print_dataset_sizes(all_datasets)

    params = [p for p in model.parameters() if p.requires_grad]
    image_encoder = ImageClassifier.load(args.load).image_encoder if args.freeze_encoder else None
    loss_fn, optimizer = get_training_components(params, args)
    dataloaders = prepare_dataloaders(all_datasets, args, image_encoder)

    if args.method == "autoft" and args.k is not None:
        fs_id_dataset, fs_val_dataset = all_datasets["id"], all_datasets["id_val"]
        return auto_ft(args, model, dataloaders, all_datasets["ood_subset_for_hp"], args.autoft_epochs, input_key, fs_id_dataset, fs_val_dataset)

    return finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every=100)


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
    print(f"\nSaved evaluation results to {results_path}.")
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
    main(args)