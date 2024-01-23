import src.datasets as datasets
import torch
from src.datasets.common import get_dataloader
from src.datasets.laion import get_data
from src.models.utils import get_num_classes


def extract_fewshot_samples(iterator, args, id_train=True):
    samples = {label: {"images": [], "labels": []} for label in range(2)} if id_train else {}
    num_classes_collected = 0

    while True:
        images, labels, *texts = next(iterator)
        texts = texts[0] if texts else None

        for i, label in enumerate(labels):
            label_item = label.item()
            if id_train:
                key = 0 if (texts and torch.equal(texts[0], texts[i])) else 1
            else:
                if label_item not in samples:
                    samples[label_item] = {"images": [], "labels": []}
                key = label_item

            if len(samples[key]["images"]) < args.k:
                samples[key]["images"].append(images[i])
                samples[key]["labels"].append(labels[i])

            if len(samples[key]["images"]) == args.k:
                num_classes_collected += 1
                if num_classes_collected == (2 if id_train else get_num_classes(args)):
                    break

    final_images, final_labels = [], []
    for data in samples.values():
        final_images.extend(data["images"][:args.k])
        final_labels.extend(data["labels"][:args.k])

    return torch.stack(final_images, dim=0), torch.stack(final_labels, dim=0)


def get_id_dataloader(args, model, preprocess_fn):
    if args.ft_data is not None:
        train_preprocess_fn, val_preprocess_fn = get_preprocess_fns(model)
        img_text_data = get_data(args, (train_preprocess_fn, val_preprocess_fn), epoch=0)
        return img_text_data['train_ft'].dataloader
    else:
        id_dataset_class = getattr(datasets, args.id)
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location)
        return get_dataloader(id_dataset, is_train=True, args=args, image_encoder=None)


def get_val_dataset(args, preprocess_fn):
    val_dataset_class = getattr(datasets, args.id)
    val_dataset = val_dataset_class(preprocess_fn, location=args.data_location, batch_size=args.k)
    val_dataloader = get_dataloader(val_dataset, is_train=False, args=args, image_encoder=None)
    return extract_fewshot_samples(iter(val_dataloader), args, id_train=False)


def get_ood_dataset(args, preprocess_fn):
    ood_dataset_class = getattr(datasets, args.ood)
    ood_dataset = ood_dataset_class(preprocess=preprocess_fn, train=True, location=args.data_location, batch_size=args.batch_size)
    ood_dataloader = get_dataloader(ood_dataset, is_train=True, args=args, image_encoder=None)
    return extract_fewshot_samples(iter(ood_dataloader), args, id_train=False)


def get_fewshot_datasets(args, model, preprocess_fn):
    orig_batch_size = args.batch_size
    args.batch_size = args.k

    id_dataloader = get_id_dataloader(args, model, preprocess_fn)
    id_image, id_label, id_text = extract_fewshot_samples(iter(id_dataloader), args, id_train=True)

    val_image, val_label = get_val_dataset(args, preprocess_fn)
    ood_image, ood_label = get_ood_dataset(args, preprocess_fn)

    args.batch_size = orig_batch_size
    return {
        "id": (id_image, id_label, id_text),
        "id_val": (val_image, val_label),
        "ood_subset_for_hp": (ood_image, ood_label)
    }

def get_preprocess_fns(model):
    return model.module.image_encoder.train_preprocess, model.module.image_encoder.val_preprocess


def get_standard_datasets(args, model, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    if args.ft_data is not None:
        train_preprocess_fn, val_preprocess_fn = get_preprocess_fns(model)
        id_dataset = get_data(args, (train_preprocess_fn, val_preprocess_fn), epoch=0)
    else:
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, n_examples=args.num_id_examples, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    id_val_dataset = id_dataset_class(preprocess=preprocess_fn, train=False, n_examples=args.num_id_examples, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    return id_dataset, id_val_dataset


def get_datasets(args, model, preprocess_fn):
    if args.k is not None:
        return get_fewshot_datasets(args, model, preprocess_fn)

    id_dataset, id_val_dataset = get_standard_datasets(args, model, preprocess_fn)
    ood_subset_for_hp = get_ood_dataset(args, preprocess_fn, standard=True)

    all_datasets = {"id": id_dataset, "id_val": id_val_dataset, "ood_subset_for_hp": ood_subset_for_hp}
    return all_datasets