import torch
import src.datasets as datasets
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.laion import get_dataset_fn, DataInfo
from src.models import utils


def eval_single_dataset(image_classifier: torch.nn.Module, classification_head: torch.nn.Module,
                        dataset: DataInfo, args) -> dict:
    """Evaluates a single dataset using the given image classifier and classification head."""
    image_encoder = image_classifier.module.image_encoder if args.freeze_encoder else None
    model = image_classifier.cuda()
    classification_head = classification_head.cuda()

    model.eval()
    classification_head.eval()

    dataloader = dataset.dataloader if isinstance(dataset, DataInfo) else get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_encoder)

    metrics = {}
    correct, total_samples = 0, 0
    all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        for data in dataloader:
            data = maybe_dictionarize(data)
            x, y = data['images'].cuda(), data['labels'].cuda()
            image_paths = data.get('image_paths')

            logits = utils.get_logits_encoder(x, model.module.image_encoder, classification_head)
            logits = dataset.project_logits(logits, x.device) if hasattr(dataset, 'project_logits') else logits
            y = dataset.project_labels(y, x.device) if hasattr(dataset, 'project_labels') else y

            pred = logits.argmax(dim=1, keepdim=True).to(x.device)
            acc, total = dataset.accuracy(logits, y, image_paths, args) if hasattr(dataset, 'accuracy') else (
                pred.eq(y.view_as(pred)).sum().item(), y.size(0))
            correct += acc
            total_samples += total

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().detach())
                all_preds.append(logits.cpu().detach())
                metadata = data.get('metadata', image_paths)
                all_metadata.extend(metadata)

        metrics['top1'] = correct / total_samples if total_samples > 0 else 0

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics.update(dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args))

    torch.cuda.empty_cache()
    return metrics


def eval_single_batch_dataset(image_classifier: torch.nn.Module, dataset: DataInfo, args,
                              classification_head: torch.nn.Module, data) -> (float, float):
    """Evaluates a single batch of data from the dataset using the image classifier and classification head."""
    model = image_classifier
    model.eval()
    classification_head.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        top1_accuracy, total_correct, total_samples, total_loss = 0.0, 0, 0, 0.0

        data = maybe_dictionarize(data)
        x, y = data['images'].to(device), data['labels'].to(device)
        assert x.shape[0] == 2 * args.k, 'Validation batch size mismatch.'

        image_paths = data.get('image_paths')
        logits = utils.get_logits_encoder(x, model.module.image_encoder, classification_head)

        if hasattr(dataset, 'project_logits'):
            logits = dataset.project_logits(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        loss = torch.nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)

        if hasattr(dataset, 'accuracy'):
            acc, num_samples = dataset.accuracy(logits, y, image_paths, args)
            total_correct += acc
            total_samples += num_samples
        else:
            total_correct += pred.eq(y.view_as(pred)).sum().item()
            total_samples += y.size(0)

        top1_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        if hasattr(dataset, 'post_loop_metrics'):
            # Collecting labels, predictions, and metadata for post-loop metrics
            all_labels = [y.cpu().clone().detach()]
            all_preds = [logits.cpu().clone().detach()]
            all_metadata = [data.get('metadata', image_paths)]

            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {'top1': top1_accuracy}

    return metrics['top1'], loss.item()


def evaluate(image_classifier, classification_head, args):
    """Evaluate the model on multiple datasets."""
    if args.eval_datasets is None:
        return {}

    preprocess_fn = image_classifier.module.val_preprocess
    info = vars(args)
    eval_datasets = args.eval_datasets

    for dataset_name in eval_datasets:
        print(f"Evaluating on {dataset_name}")
        dataset_class = getattr(datasets, dataset_name)
        if dataset_name in ["ImageNet4", "ImageNet16", "ImageNet32"]:
            dataset = (get_dataset_fn(args.ft_data, args.dataset_type)(args, preprocess_fn, is_train=False, epoch=0))
        else:
            dataset = dataset_class(preprocess_fn, n_examples=-1, location=args.data_location,
                                    batch_size=args.batch_size)

        print("Loaded dataset")
        results = eval_single_dataset(image_classifier, classification_head, dataset, args)
        torch.cuda.empty_cache()
        print(f"{dataset_name} Top-1 accuracy: {results.get('top1', 0):.4f}")

        for key, val in results.items():
            if key in ['worst', 'f1', 'pm0']:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[f"{dataset_name}:{key}"] = val

    return info
