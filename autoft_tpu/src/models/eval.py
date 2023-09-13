import json
import logging
import os

import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils
from src.models.utils import is_tpu_available

logger = logging.getLogger('main')

def _move_model_to_device(image_classifier, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None
    device = xm.xla_device()
    return model.to(device), input_key, image_enc


def _calculate_accuracy(dataset, logits, y, image_paths, args):
    if hasattr(dataset, 'accuracy'):
        acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
    else:
        pred = logits.argmax(dim=1, keepdim=True)
        acc1 = pred.eq(y.view_as(pred)).sum().item()
        num_total = y.size(0)
    return acc1, num_total


def _process_batch(data, dataset, model, input_key, args):
    """Process batch and return results."""
    x = data[input_key]
    y = data['labels']

    image_paths = data.get('image_paths', None)
    logits = utils.get_logits(x, model)

    projection_fn = getattr(dataset, 'project_logits', None)
    device = xm.xla_device()
    if projection_fn:
        logits = projection_fn(logits, device)
    if hasattr(dataset, 'project_labels'):
        y = dataset.project_labels(y, device)

    acc1, num_total = _calculate_accuracy(dataset, logits, y, image_paths, args)

    if hasattr(dataset, 'post_loop_metrics'):
        all_labels = y.cpu().clone().detach()
        all_preds = logits.cpu().clone().detach()
        metadata = data.get('metadata', image_paths)
        return acc1, num_total, all_labels, all_preds, metadata

    return acc1, num_total, None, None, None


def eval_single_dataset(image_classifier, dataset, args):
    """Evaluate a single dataset and return metrics."""
    model, input_key, image_enc = _move_model_to_device(image_classifier, args)
    model.eval()
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_enc)

    top1, correct, n = 0., 0., 0.
    all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            acc1, batch_size, labels, preds, metadata = _process_batch(data, dataset, model, input_key, args)
            correct += acc1
            n += batch_size

            if labels is not None:
                all_labels.append(labels)
                all_preds.append(preds)
                all_metadata.extend(metadata)


    top1 = correct / n

    metrics = {}
    if hasattr(dataset, 'post_loop_metrics'):
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
        if 'acc' in metrics:
            metrics['top1'] = metrics['acc']
    if 'top1' not in metrics:
        metrics['top1'] = top1

    if is_tpu_available():
        metrics = xm.mesh_reduce('metrics_reduce', metrics, lambda x: {k: sum(v) for k, v in zip(x.keys(), x.values())})

    return metrics


def _mp_evaluate(rank, image_classifier, args, spawn_required=True):
    """Evaluate on multiple datasets and print results."""
    if args.eval_datasets is None:
        return
    info = vars(args)
    for dataset_name in args.eval_datasets:
        xm.master_print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            train=False,
            n_examples=-1,
            location=args.data_location,
            batch_size=args.batch_size
        )
        results = eval_single_dataset(image_classifier, dataset, args)
        for key, val in results.items():
            prefix = f"{dataset_name} "
            if key == 'top1':
                xm.master_print(f"{prefix}Top-1 accuracy: {val:.4f}")
            elif 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                xm.master_print(f"{prefix}{key}: {val:.4f}")
            info[f"{dataset_name}:{key}"] = val

    xm.master_print(info)
    if xm.is_master_ordinal():
        logger.info(json.dumps(info, indent=4))
        os.makedirs(args.save, exist_ok=True)
        results_path = os.path.join(args.save, 'eval_results.json')
        with open(results_path, 'w') as f:
            f.write(json.dumps(info))
        xm.master_print(f'\nSaved evaluation results to {results_path}.')


def evaluate(image_classifier, args, spawn_required=True):
    """Depending on the flag, either spawn new processes or directly evaluate."""
    if spawn_required:
        # If called outside of the training loop, spawn new processes.
        xmp.spawn(_mp_evaluate, args=(image_classifier, args,), nprocs=8, start_method='spawn')
    else:
        # If called within the training loop, use the current process.
        rank = xm.get_ordinal()
        _mp_evaluate(rank, image_classifier, args)