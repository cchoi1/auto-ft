import src.datasets as datasets
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils
from src.models.utils import get_device, is_tpu_available


def _move_model_to_device(image_classifier, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None
    return model.to(get_device()), input_key, image_enc


def _prepare_dataloader(dataset, args, image_enc=None):
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_enc)
    if is_tpu_available():
        dataloader = pl.MpDeviceLoader(dataloader, get_device())
    return dataloader


def _calculate_accuracy(dataset, logits, y, image_paths, args):
    if hasattr(dataset, 'accuracy'):
        acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
    else:
        pred = logits.argmax(dim=1, keepdim=True).to(get_device())
        acc1 = pred.eq(y.view_as(pred)).sum().item()
        num_total = y.size(0)
    return acc1, num_total


def _process_batch(data, dataset, model, input_key, args):
    """Process batch and return results."""
    device = get_device()
    x = data[input_key].to(device)
    y = data['labels'].to(device)

    image_paths = data.get('image_paths', None)
    logits = utils.get_logits(x, model)

    projection_fn = getattr(dataset, 'project_logits', None)
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
    dataloader = _prepare_dataloader(dataset, args, image_enc)

    top1, correct, n = 0., 0., 0.
    all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            acc1, num_total, labels, preds, metadata = _process_batch(data, dataset, model, input_key, args)
            correct += acc1
            n += num_total

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


def evaluate(image_classifier, args):
    """Evaluate on multiple datasets and print results."""
    if args.eval_datasets is None:
        return
    info = vars(args)
    for dataset_name in args.eval_datasets:
        print('Evaluating on', dataset_name)
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
                print(f"{prefix}Top-1 accuracy: {val:.4f}")
            elif 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{prefix}{key}: {val:.4f}")
            info[f"{dataset_name}:{key}"] = val
    return info



# eval.py from FLYP codebase which uses classification head
#
# import osx
# import json
#
# import torch
# import numpy as np
# from src.models import utils
# from src.datasets.common import get_dataloader, maybe_dictionarize
# import src.datasets as datasets
# import torch.nn.functional as F
#
#
# def eval_single_dataset(image_classifier, dataset, args, classification_head):
#
#     model = image_classifier
#     input_key = 'images'
#     image_enc = None
#
#     model.eval()
#     classification_head.eval()
#
#     dataloader = get_dataloader(dataset,
#                                 is_train=False,
#                                 args=args,
#                                 image_encoder=image_enc)
#
#     batched_data = enumerate(dataloader)
#     device = args.device
#
#     if hasattr(dataset, 'post_loop_metrics'):
#         # keep track of labels, predictions and metadata
#         all_labels, all_preds, all_metadata = [], [], []
#
#     with torch.no_grad():
#         top1, correct, n = 0., 0., 0.
#         for i, data in batched_data:
#
#             data = maybe_dictionarize(data)
#             x = data[input_key].to(device)
#             y = data['labels'].to(device)
#
#             if 'image_paths' in data:
#                 image_paths = data['image_paths']
#
#             logits = utils.get_logits(x, model, classification_head)
#
#             projection_fn = getattr(dataset, 'project_logits', None)
#             if projection_fn is not None:
#                 logits = projection_fn(logits, device)
#
#             if hasattr(dataset, 'project_labels'):
#                 y = dataset.project_labels(y, device)
#             pred = logits.argmax(dim=1, keepdim=True).to(device)
#             if hasattr(dataset, 'accuracy'):
#                 acc1, num_total = dataset.accuracy(logits, y, image_paths,
#                                                    args)
#                 correct += acc1
#                 n += num_total
#             else:
#                 correct += pred.eq(y.view_as(pred)).sum().item()
#                 n += y.size(0)
#
#             if hasattr(dataset, 'post_loop_metrics'):
#                 all_labels.append(y.cpu().clone().detach())
#                 all_preds.append(logits.cpu().clone().detach())
#                 metadata = data[
#                     'metadata'] if 'metadata' in data else image_paths
#                 all_metadata.extend(metadata)
#
#         top1 = correct / n
#
#         if hasattr(dataset, 'post_loop_metrics'):
#             all_labels = torch.cat(all_labels)
#             all_preds = torch.cat(all_preds)
#             metrics = dataset.post_loop_metrics(all_labels, all_preds,
#                                                 all_metadata, args)
#             if 'acc' in metrics:
#                 metrics['top1'] = metrics['acc']
#         else:
#             metrics = {}
#     if 'top1' not in metrics:
#         metrics['top1'] = top1
#
#     return metrics
#
#
# def eval_single_batch_dataset(image_classifier, dataset, args,
#                               classification_head, data):
#
#     model = image_classifier
#     input_key = 'images'
#
#     model.eval()
#     classification_head.eval()
#
#     device = args.device
#
#     if hasattr(dataset, 'post_loop_metrics'):
#         # keep track of labels, predictions and metadata
#         all_labels, all_preds, all_metadata = [], [], []
#
#     with torch.no_grad():
#         top1, correct, n, cnt_loss = 0., 0., 0., 0.
#
#         data = maybe_dictionarize(data)
#         x = data[input_key].to(device)
#         y = data['labels'].to(device)
#
#         assert x.shape[0] == 2 * args.k, 'val mismatch size'
#
#         if 'image_paths' in data:
#             image_paths = data['image_paths']
#
#         logits = utils.get_logits(x, model, classification_head)
#
#         projection_fn = getattr(dataset, 'project_logits', None)
#         if projection_fn is not None:
#             logits = projection_fn(logits, device)
#
#         if hasattr(dataset, 'project_labels'):
#             y = dataset.project_labels(y, device)
#
#         cnt_loss = F.cross_entropy(logits, y)
#         pred = logits.argmax(dim=1, keepdim=True).to(device)
#         if hasattr(dataset, 'accuracy'):
#             acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
#             correct += acc1
#             n += num_total
#         else:
#             correct += pred.eq(y.view_as(pred)).sum().item()
#             n += y.size(0)
#
#         if hasattr(dataset, 'post_loop_metrics'):
#             all_labels.append(y.cpu().clone().detach())
#             all_preds.append(logits.cpu().clone().detach())
#             metadata = data['metadata'] if 'metadata' in data else image_paths
#             all_metadata.extend(metadata)
#
#         top1 = correct / n
#
#         if hasattr(dataset, 'post_loop_metrics'):
#             all_labels = torch.cat(all_labels)
#             all_preds = torch.cat(all_preds)
#             metrics = dataset.post_loop_metrics(all_labels, all_preds,
#                                                 all_metadata, args)
#             if 'acc' in metrics:
#                 metrics['top1'] = metrics['acc']
#         else:
#             metrics = {}
#     if 'top1' not in metrics:
#         metrics['top1'] = top1
#
#     return metrics['top1'], cnt_loss.item()
#
#
# def evaluate(image_classifier,
#              args,
#              classification_head,
#              train_stats={},
#              logger=None):
#     if args.eval_datasets is None:
#         return
#     info = vars(args)
#     for i, dataset_name in enumerate(args.eval_datasets):
#         print('Evaluating on', dataset_name)
#         dataset_class = getattr(datasets, dataset_name)
#         dataset = dataset_class(image_classifier.module.val_preprocess,
#                                 location=args.data_location,
#                                 batch_size=args.batch_size)
#
#         results = eval_single_dataset(image_classifier, dataset, args,
#                                       classification_head)
#
#         if 'top1' in results:
#             print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
#             if logger != None:
#                 logger.info(
#                     f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
#             train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)
#
#         for key, val in results.items():
#             if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
#                 print(f"{dataset_name} {key}: {val:.4f}")
#                 if logger != None:
#                     logger.info(f"{dataset_name} {key}: {val:.4f}")
#                 train_stats[dataset_name + key] = round(val, 4)
#
#     return info