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

    model = model.cuda()
    model.eval()
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].cuda()
            y = data['labels'].cuda()
            device = x.device

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
        torch.cuda.empty_cache()
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics


def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    if isinstance(image_classifier, torch.nn.DataParallel):
        preprocess_fn = image_classifier.module.val_preprocess
    else:
        preprocess_fn = image_classifier.val_preprocess
    info = vars(args)
    eval_datasets = args.eval_datasets
    for i, dataset_name in enumerate(eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            preprocess_fn,
            train=False,
            n_examples=-1,
            location=args.data_location,
            batch_size=args.batch_size
        )
        print('loaded dataset')
        results = eval_single_dataset(image_classifier, dataset, args)
        torch.cuda.empty_cache()
        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    return info