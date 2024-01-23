import copy
import logging
import time

import torch
from src.datasets.common import maybe_dictionarize
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate, eval_single_batch_dataset
from src.models.utils import cosine_lr, extract_from_data_parallel, val_metric_str, print_train_update
from src.models.zeroshot import get_zeroshot_classifier

logger = logging.getLogger('main')


def compute_accuracy(model, dataloader, input_key):
    """Compute the accuracy of a model on a given dataloader."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs, labels = batch[input_key].cuda(), batch['labels'].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total


def calculate_loss(loss_fn, model, logits, labels, image_features, text_features, logit_scale, args):
    """Calculate the loss for a given batch."""
    ls = logit_scale[0] if isinstance(logit_scale, list) else logit_scale
    if isinstance(loss_fn, LearnedLoss):
        loss = loss_fn(model, logits, labels, image_features, text_features, ls)
    else:
        loss = loss_fn(logits, labels)

    return loss / args.accumulation_steps


def get_batch(dataloaders, args, fs_id_dataset=None):
    """Get a batch from either few-shot or standard dataset."""
    if fs_id_dataset:
        images, labels = fs_id_dataset
        text = None
    else:
        batch = maybe_dictionarize(next(iter(dataloaders["id"])))
        images, labels = batch[args.input_key].cuda(), batch['labels'].cuda()
        text = batch['metadata'].cuda() if 'metadata' in batch and args.ft_data else None

    return images, labels, text


@torch.no_grad()
def evaluate_net(net, dataloader, dataset, args, is_fewshot=False):
    """Evaluate the network on a given dataset."""
    total_correct, total_samples = 0, 0
    all_labels, all_preds, all_metadata = [], [], []

    net = net.cuda().eval()
    if args.regenerate_head:
        net.module.classification_head = get_zeroshot_classifier(args, net.module.image_encoder.model)

    if is_fewshot:
        dataloader = [(dataset[0], dataset[1])]

    for batch in dataloader:
        data = maybe_dictionarize(batch, is_fewshot)
        x, y = data['images'].cuda(), data['labels'].cuda()
        outputs = net(x)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == y).sum().item()
        total_samples += y.size(0)
        all_labels.append(y.cpu())
        all_preds.append(outputs.cpu())
        if 'metadata' in data:
            all_metadata.extend(data['metadata'])

    accuracy = total_correct / total_samples
    all_labels, all_preds = torch.cat(all_labels), torch.cat(all_preds)
    xent = torch.nn.CrossEntropyLoss(reduction='none')(all_preds, all_labels).mean().item()

    metrics = {'acc': accuracy * 100, "xent": xent}
    if hasattr(dataset, 'post_loop_metrics'):
        metrics.update(dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args))

    return metrics


def inner_finetune(args, model, loss_fn, optimizer, dataloaders, ood_hp_dataset, fs_id_dataset=None,
                   fs_val_dataset=None):
    """Inner finetuning loop."""
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length * args.accumulation_steps, args.inner_steps)
    model.train()
    val_metrics = {}

    for step in range(args.inner_steps * args.accumulation_steps):
        scheduler(step)
        images, labels, text = get_batch(dataloaders, args, fs_id_dataset)
        logits, image_features, text_features, logit_scale = model(images.cuda(), text.cuda())
        loss = calculate_loss(loss_fn, model, logits, labels, image_features, text_features, logit_scale, args)
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step + 1 in args.inner_loop_val_steps or step == args.inner_steps * args.accumulation_steps - 1:
            val_dataset = fs_val_dataset if fs_id_dataset else ood_hp_dataset
            val_metrics[step + 1] = evaluate_net(model, dataloaders["ood_hp"], val_dataset, args,
                                                 fs_id_dataset is not None)

    return model, val_metrics


def finetune(args, model, loss_fn, optimizer, dataloaders, input_key, print_every, fs_id_dataset=None,
             fs_val_dataset=None):
    """Finetuning process for the model."""
    finetune_type = 'few-shot' if args.k is not None else 'full'
    print(f"Finetuning with {finetune_type} data.")
    assert fs_id_dataset is not None and fs_val_dataset is not None if finetune_type == 'few-shot' else True

    num_batches = 1 if finetune_type == 'few-shot' else len(dataloaders["id"])
    total_steps = args.ft_epochs * num_batches
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length * args.accumulation_steps, total_steps)
    model_copy = copy.deepcopy(model.module).cpu()
    best_metric = float('inf') if finetune_type == 'few-shot' else -float('inf')
    val_metrics = {}

    for epoch in range(args.ft_epochs):
        model.train()

        for i in range(num_batches):
            step = i + epoch * num_batches
            scheduler(step)
            images, labels, text = get_batch(dataloaders, args, fs_id_dataset if finetune_type == 'few-shot' else None)
            logits, image_features, text_features, logit_scale = model(images.cuda(), text.cuda() if text else None)
            loss = calculate_loss(loss_fn, model, logits, labels, image_features, text_features, logit_scale, args)
            loss.backward()

            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Print training update for full finetuning
            if finetune_type == 'full':
                batch_time = time.time() - start_time
                print_train_update(logger, print_every, total_steps, step, loss, batch_time)

        # Evaluate the model
        if finetune_type == 'full':
            eval_results = evaluate(model, model.module.classification_head, args)
            val_metrics[epoch] = eval_results
            curr_metric = eval_results[val_metric_str(args)]
        else:
            with torch.no_grad():
                model.eval()
                classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model).cuda()
                eval_results = eval_single_batch_dataset(model, dataloaders["id_val"].dataset, args,
                                                         classification_head, fs_val_dataset)
                curr_metric = eval_results[1]  # Assuming val_loss

        # Update the best metric and model copy
        is_best_metric = curr_metric < best_metric if finetune_type == 'few-shot' else curr_metric > best_metric
        if is_best_metric:
            best_metric = curr_metric
            model_copy = copy.deepcopy(extract_from_data_parallel(model)).cpu()
            for param in model_copy.parameters():
                param.requires_grad = False

    # Clean up and prepare the final model
    del model
    torch.cuda.empty_cache()
    model_copy = torch.nn.DataParallel(model_copy.cuda(), device_ids=list(range(torch.cuda.device_count())))

    return model_copy, best_metric
