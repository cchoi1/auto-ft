import json
import os

import src.datasets as datasets
import torch
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils


def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None
    print('freeze encoder', args.freeze_encoder)
    print('dataset', str(dataset))
    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics


def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    eval_datasets = args.eval_datasets
    for i, dataset_name in enumerate(eval_datasets):
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

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

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