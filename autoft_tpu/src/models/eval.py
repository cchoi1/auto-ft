import json
import logging
import os
import time

import numpy as np
import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils

logger = logging.getLogger('main')

def _calculate_accuracy(dataset, logits, y, image_paths, args):
    if hasattr(dataset, 'accuracy'):
        acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
    else:
        pred = logits.argmax(dim=1, keepdim=True)
        acc1 = pred.eq(y.view_as(pred)).sum().item()
        num_total = y.size(0)
    return acc1, num_total


def _process_batch(batch, dataset, image_encoder, classification_head, input_key, args):
    """Process batch and return results."""
    batch = maybe_dictionarize(batch)
    device = xm.xla_device()
    x = batch[input_key].to(device)
    y = batch['labels'].to(device)

    image_paths = batch.get('image_paths', None)
    # logits = utils.get_logits(x, model)
    logits = utils.get_logits_encoder(x, image_encoder, classification_head)

    projection_fn = getattr(dataset, 'project_logits', None)
    if projection_fn:
        logits = projection_fn(logits, device)
    if hasattr(dataset, 'project_labels'):
        y = dataset.project_labels(y, device)

    acc1, num_total = _calculate_accuracy(dataset, logits, y, image_paths, args)

    if hasattr(dataset, 'post_loop_metrics'):
        all_labels = y.cpu().clone().detach()
        all_preds = logits.cpu().clone().detach()
        metadata = batch.get('metadata', image_paths)
        return acc1, num_total, all_labels, all_preds, metadata

    return acc1, num_total, None, None, None


def eval_single_dataset(image_classifier, classification_head, dataset, args):
    """Evaluate a single dataset and return metrics."""
    device = xm.xla_device()
    image_encoder = image_classifier.image_encoder
    image_encoder.to(device)
    image_encoder.eval()
    image_classifier.to(device)
    image_classifier.eval()
    input_key = "images"
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_encoder)

    top1, correct, n = 0., 0., 0.
    all_labels, all_preds, all_metadata = [], [], []
    xm.master_print(f"Num batches in eval dataloader: {len(dataloader)}")
    with torch.no_grad():
        for batch in dataloader:
            batch_start_time = time.time()
            acc1, batch_size, labels, preds, metadata = _process_batch(batch, dataset, image_encoder, input_key, args)
            correct += acc1
            n += batch_size

            if labels is not None:
                all_labels.append(labels)
                all_preds.append(preds)
                all_metadata.extend(metadata)
            xm.master_print(f"Batch took {time.time() - batch_start_time} seconds")

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
    for k,v in metrics.items():
        metrics[k] = xm.mesh_reduce(f"{k}", v, np.mean)

    return metrics


def _mp_evaluate(rank, image_classifier, classification_head, args):
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
        results = eval_single_dataset(image_classifier, classification_head, dataset, args)
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


def evaluate(image_classifier, classification_head, args, spawn_required=True):
    """Depending on the flag, either spawn new processes or directly evaluate."""
    if spawn_required:
        # If called outside of the training loop, spawn new processes.
        xmp.spawn(_mp_evaluate, args=(image_classifier, classification_head, args,), nprocs=8, start_method='spawn')
    else:
        # If called within the training loop, use the current process.
        rank = xm.get_ordinal()
        _mp_evaluate(rank, image_classifier, classification_head, args)