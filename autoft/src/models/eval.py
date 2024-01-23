import torch
import src.datasets as datasets
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.laion import get_dataset_fn, DataInfo
from src.models import utils

def eval_single_dataset(image_classifier: torch.nn.Module, classification_head: torch.nn.Module,
                        dataset: DataInfo, args) -> dict:
    input_key = 'features' if args.freeze_encoder else 'images'
    image_encoder = image_classifier.module.image_encoder if args.freeze_encoder else None

    model = image_classifier.cuda()
    classification_head = classification_head.cuda()
    model.eval()
    classification_head.eval()

    dataloader = dataset.dataloader if isinstance(dataset, DataInfo) else get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_encoder)

    metrics = {}
    with torch.no_grad():
        correct, total_samples = 0, 0
        all_labels, all_preds, all_metadata = [], [], []

        for i, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x, y = data[input_key].cuda(), data['labels'].cuda()
            image_paths = data.get('image_paths')

            logits = utils.get_logits_encoder(x, model.module.image_encoder, classification_head)
            logits = dataset.project_logits(logits, x.device) if hasattr(dataset, 'project_logits') else logits
            y = dataset.project_labels(y, x.device) if hasattr(dataset, 'project_labels') else y

            pred = logits.argmax(dim=1, keepdim=True).to(x.device)
            if hasattr(dataset, 'accuracy'):
                acc, total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc
                total_samples += total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_samples += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().detach())
                all_preds.append(logits.cpu().detach())
                metadata = data.get('metadata', image_paths)
                all_metadata.extend(metadata)

        top1_accuracy = correct / total_samples
        metrics['top1'] = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)['acc'] \
            if hasattr(dataset, 'post_loop_metrics') else top1_accuracy

    torch.cuda.empty_cache()
    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args, classification_head, data):
    model = image_classifier
    input_key = 'images'
    model.eval()
    classification_head.eval()
    device = next(model.parameters()).device

    if hasattr(dataset, 'post_loop_metrics'):
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.
        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)
        assert x.shape[0] == 2 * args.k, 'val mismatch size'
        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits_encoder(x, image_classifier.module.image_encoder, classification_head)
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)
        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = torch.nn.functional.cross_entropy(logits, y)
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

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier, classification_head, args):
    if args.eval_datasets is None:
        return
    preprocess_fn = image_classifier.module.val_preprocess
    info = vars(args)
    eval_datasets = args.eval_datasets
    for i, dataset_name in enumerate(eval_datasets):
        print(f"Evaluating on {dataset_name}")
        dataset_class = getattr(datasets, dataset_name)
        if dataset_name in ["ImageNet4", "ImageNet16", "ImageNet32"]:
            val_preprocess_fn = image_classifier.module.image_encoder.val_preprocess
            dataset = get_dataset_fn(args.ft_data, args.dataset_type)(args, val_preprocess_fn, is_train=False, epoch=0)
        else:
            dataset = dataset_class(
                preprocess_fn,
                train=False,
                n_examples=-1,
                location=args.data_location,
                batch_size=args.batch_size
            )
        print("Loaded dataset")
        results = eval_single_dataset(image_classifier, classification_head, dataset, args)
        torch.cuda.empty_cache()
        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    return info