import time
import logging

import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.utils import get_ood_datasets
from src.logger import setup_logging
from src.models.autoft import auto_ft
from src.models.eval import evaluate
from src.models.finetune import finetune_final
from src.models.modeling import ImageClassifier
from src.models.networks import get_pretrained_net_fixed
import src.datasets as datasets
from torchvision import transforms

def initialize_model(args):
    if args.model == "svhn":
        model = get_pretrained_net_fixed(ckpt_path="/iris/u/cchoi1/robust-optimizer/mnist/ckpts", dataset_name="svhn", output_channels=3, train=True)
        preprocess_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    else:
        image_classifier = ImageClassifier.load(args.load)
        if args.freeze_encoder:
            model = image_classifier.classification_head
            preprocess_fn = image_classifier.val_preprocess
        else:
            model = image_classifier
            preprocess_fn = image_classifier.train_preprocess
            image_classifier.process_images = True
    devices = list(range(torch.cuda.device_count()))
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model, preprocess_fn

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


def run_method(args, model, preprocess_fn):
    if args.eval_only:
        return evaluate(model, args)
    if args.freeze_encoder:
        input_key = 'features'
        print_every = 1000
    else:
        input_key = 'images'
        print_every = 100
    all_datasets = get_datasets(args, preprocess_fn)
    print("Got datasets")
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

def main(args, rank=0):
    logger = logging.getLogger('main')
    logger = setup_logging(args, logger)
    logger.info(args)
    model, preprocess_fn = initialize_model(args)
    run_method(args, model, preprocess_fn)


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    main(args)

    print(f"\nRUN TIME: {time.time() - start_time:.3f}", flush=True)
