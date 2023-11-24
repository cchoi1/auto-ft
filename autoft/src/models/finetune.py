import copy
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
from src.datasets.common import maybe_dictionarize, get_dataloader
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate, eval_single_batch_dataset
from src.models.modeling import ImageClassifier
from src.models.utils import cosine_lr, extract_from_data_parallel, val_metric_str, test_metric_str
from src.models.zeroshot import get_zeroshot_classifier

logger = logging.getLogger('main')

def print_train_update(logger, print_every, total_steps, step, loss, batch_time):
    should_print = (print_every is not None and step % print_every == 0)
    if should_print:
        percent_complete = 100 * step / total_steps
        print(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
            f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}")
        logger.info(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                    f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}")

def save_finetuned_model(args, model, optimizer, logger):
    os.makedirs(args.save, exist_ok=True)
    model_path = os.path.join(args.save, f'checkpoint_{args.ft_epochs}.pt')
    print('Saving model to', model_path)
    logger.info(f"Saving model to {model_path}")
    model = extract_from_data_parallel(model)
    model.save(model_path)
    optim_path = os.path.join(args.save, f'optim_{args.ft_epochs}.pt')
    torch.save(optimizer.state_dict(), optim_path)
    print('Saving optimizer to', optim_path)
    logger.info(f"Saving optimizer to {optim_path}")
    return model_path


def compute_accuracy(model, dataloader, input_key):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set the model back to train mode
    return correct / total


@torch.no_grad()
def evaluate_net(net, dataloader, dataset, args):
    total_correct = 0
    total_samples = 0
    all_labels, all_preds, all_metadata = [], [], []

    if args.regenerate_head:
        net.module.classification_head = get_zeroshot_classifier(args, net.module.image_encoder.model)

    net = net.cuda()
    net.eval()
    for batch in dataloader:

        data = maybe_dictionarize(batch)
        x = data['images'].cuda()
        y = data['labels'].cuda()
        outputs, _, _, _ = net(x)

        if (y == -1).any():  # Handle unlabeled parts of the batch
            pseudo_labels = torch.argmax(outputs, dim=1)
            mask = (y == -1)
            y[mask] = pseudo_labels[mask]

        predictions = outputs.argmax(dim=1)
        correct = (predictions == y).sum().item()
        total_correct += correct
        total_samples += y.size(0)
        all_labels.append(y.cpu().clone().detach())
        all_preds.append(outputs.cpu().clone().detach())
        if 'metadata' in data:
            all_metadata.extend(data['metadata'])

    accuracy = (total_correct / total_samples) * 100
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    xent_fn = torch.nn.CrossEntropyLoss(reduction='none')
    xent = xent_fn(all_preds, all_labels).cpu().numpy().mean()
    xent = float(xent)

    # Calculate post loop metrics if available
    if hasattr(dataset, 'post_loop_metrics'):
        metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
        if 'acc' not in metrics:
            metrics['acc'] = accuracy
        if "xent" not in metrics:
            metrics["xent"] = xent
    else:
        metrics = {'acc': accuracy, "xent": xent}

    torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def evaluate_net_fewshot(net, dataset, args):
    total_correct = 0
    total_samples = 0
    all_labels, all_preds, all_metadata = [], [], []

    net = net.cuda()
    net.eval()
    x, y = dataset
    x, y = x.cuda(), y.cuda()
    outputs, _, _, _ = net(x)
    predictions = outputs.argmax(dim=1)
    correct = (predictions == y).sum().item()
    total_correct += correct
    total_samples += y.size(0)
    all_labels.append(y.cpu().clone().detach())
    all_preds.append(outputs.cpu().clone().detach())

    accuracy = (total_correct / total_samples) * 100
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    xent_fn = torch.nn.CrossEntropyLoss(reduction='none')
    xent = xent_fn(all_preds, all_labels).cpu().numpy().mean()
    xent = float(xent)

    metrics = {'acc': accuracy, "xent": xent}

    torch.cuda.empty_cache()
    return metrics

def inner_finetune_fewshot(args, model, loss_fn, optimizer, ood_hp_dataset, fs_id_dataset, fs_val_dataset):
    warmup_steps = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.inner_steps)
    if len(args.inner_loop_val_steps) == 0:
        num_steps = args.inner_steps * args.accumulation_steps
    else:
        # Do up to the biggest inner loop val step; no effect if args.inner_loop_val_steps is empty
        num_steps = max(args.inner_steps, *args.inner_loop_val_steps) * args.accumulation_steps

    val_metrics = {}
    model.train()
    for step in range(num_steps):
        scheduler(step)
        images, labels, text = fs_id_dataset
        images, labels, text = images.cuda(), labels.cuda(), text.cuda()
        logits, image_features, text_features, logit_scale = model(images, text)

        if isinstance(loss_fn, LearnedLoss):
            try:
                ls = logit_scale[0]
            except Exception:
                ls = logit_scale
            loss = loss_fn(model, logits, labels, image_features, text_features, ls)
        else:
            loss = loss_fn(logits, labels)
        loss = loss / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    if args.regenerate_head:
        model.module.classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
    val_metrics["meta_objective"] = evaluate_net_fewshot(model, fs_val_dataset, args)

    return model, val_metrics


def inner_finetune_full(args, model, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset):
    torch.cuda.synchronize()
    fn_start = time.time()
    time_counter = defaultdict(list)

    if len(args.inner_loop_val_steps) == 0:
        num_steps = args.inner_steps * args.accumulation_steps
    else:
        # Do up to the biggest inner loop val step; no effect if args.inner_loop_val_steps is empty
        num_steps = max(args.inner_steps, *args.inner_loop_val_steps) * args.accumulation_steps

    val_metrics = {}
    model.train()
    dataloaders["id"].dataset.offset = np.random.randint(len(dataloaders["id"].dataset))
    for step, batch in enumerate(dataloaders["id"]):
        if step >= num_steps:
            break

        batch = maybe_dictionarize(batch)
        images, labels, text = batch[input_key].cuda(), batch['labels'].cuda(), None
        if args.ft_data is not None and 'metadata' in batch.keys():
            text = batch['metadata']
        logits, image_features, text_features, logit_scale = model(images, text)

        torch.cuda.synchronize()
        start = time.time()
        unlabeled_logits = None
        if dataloaders["unlabeled"] is not None:
            unlabeled_batch = next(iter(dataloaders["unlabeled"]))
            unlabeled_batch = maybe_dictionarize(unlabeled_batch)
            image = unlabeled_batch[input_key].cuda()
            unlabeled_logits, unlabeled_image_features, _, _ = model(image)
            pseudolabels = unlabeled_logits.argmax(dim=-1)
            unlabeled_logits = torch.cat((logits, unlabeled_logits), dim=0)
        torch.cuda.synchronize()
        time_counter["inner_step"].append(time.time() - start)

        if isinstance(loss_fn, LearnedLoss):
            try:
                ls = logit_scale[0]
            except Exception:
                ls = logit_scale
            loss = loss_fn(model, logits, labels, image_features, text_features, ls, unlabeled_logits)
        else:
            loss = loss_fn(logits, labels)
            if args.unlabeled_id is not None:
                loss += loss_fn(unlabeled_logits, pseudolabels)
        loss = loss / args.accumulation_steps

        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        time_counter["backprop"].append(time.time() - start)

        if step + 1 in args.inner_loop_val_steps:
            torch.cuda.synchronize()
            start = time.time()
            val_metrics[step + 1] = evaluate_net(model, dataloaders["ood_hp"], ood_hp_dataset, args)
            torch.cuda.synchronize()
            time_counter["eval"].append(time.time() - start)

    torch.cuda.synchronize()
    start = time.time()
    if args.regenerate_head:
        model.module.classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
    val_metrics["meta_objective"] = evaluate_net(model, dataloaders["ood_hp"], ood_hp_dataset, args)
    torch.cuda.synchronize()
    time_counter["eval"].append(time.time() - start)

    print(f"    Time per inner step: {np.mean(time_counter['inner_step']):.3f} x {len(time_counter['inner_step'])} = {sum(time_counter['inner_step']):.3f}")
    print(f"    Time per backprop: {np.mean(time_counter['backprop']):.3f} x {len(time_counter['backprop'])} = {sum(time_counter['backprop']):.3f}")
    print(f"    Time per eval: {np.mean(time_counter['eval']):.3f} x {len(time_counter['eval'])} = {sum(time_counter['eval']):.3f}")
    print(f"  Total time for inner loop: {time.time() - fn_start:.3f}")
    return model, val_metrics


def inner_finetune(args, model, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset, fs_id_dataset=None, fs_val_dataset=None):
    if fs_id_dataset is not None and fs_val_dataset is not None:
        return inner_finetune_fewshot(args, model, loss_fn, optimizer, ood_hp_dataset, fs_id_dataset, fs_val_dataset)
    else:
        return inner_finetune_full(args, model, loss_fn, optimizer, input_key, dataloaders, ood_hp_dataset)


def find_latest_checkpoint(save_dir, prefix="checkpoint_"):
    """Returns the latest checkpoint file and its epoch number based on the provided prefix."""
    checkpoints = [filename for filename in os.listdir(save_dir) if filename.startswith(prefix)]
    if not checkpoints:
        return None, -1
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.pt')[0]))
    latest_checkpoint = checkpoints[-1]
    epoch_number = int(latest_checkpoint.split('_')[1].split('.pt')[0])
    return os.path.join(save_dir, latest_checkpoint), epoch_number


def finetune_fewshot(args, model, loss_fn, optimizer, dataloaders, id_dataset, val_dataset):
    warmup_length = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, args.ft_epochs)
    min_val_loss = float('inf')
    max_val_acc = float('-inf')
    model_copy = copy.deepcopy(model.module).cpu()
    exit_flag = False
    step = 0
    for epoch in range(-1, args.ft_epochs):
        model.train()
        if epoch != -1:
            scheduler(step)

        images, labels, text = id_dataset
        images, labels, text = images.cuda(), labels.cuda(), text.cuda()
        logits, image_features, text_features, logit_scale = model(images, text)

        if isinstance(loss_fn, LearnedLoss):
            try:
                ls = logit_scale[0]
            except Exception:
                ls = logit_scale
            loss = loss_fn(model, logits, labels, image_features, text_features, ls)
        else:
            loss = loss_fn(logits, labels)
        print(f"Epoch {epoch}, Loss {loss.item()}")
        loss.backward()
        if args.clip_gradient:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Evaluate
        if epoch != -1:
            with torch.no_grad():
                model.eval()
                classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
                classification_head = classification_head.cuda()
                # val_batch = next(iter(dataloaders["id_val"]))
                eval_results = eval_single_batch_dataset(model, dataloaders["id_val"].dataset, args, classification_head, val_dataset)
                val_acc, val_loss = eval_results
                print(f"Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}")
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    max_val_acc = val_acc
                    model_copy = copy.deepcopy(extract_from_data_parallel(model)).cpu()
                    for param in model_copy.parameters():
                        param.requires_grad = False
                    print(f"Best val loss of {val_loss:.3f} and val acc of {val_acc:.3f} at epoch {epoch}.")

        if exit_flag:
            break

    model_copy = model_copy.cuda()
    model_copy = torch.nn.DataParallel(model_copy, device_ids=list(range(torch.cuda.device_count())))
    best_val_metric = min_val_loss

    return model_copy, best_val_metric


def finetune(args, model, loss_fn, optimizer, dataloaders, input_key, print_every):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    num_batches = len(dataloaders["id"])
    print(f"{num_batches} batches in fine-tuning dataloader")
    total_steps = args.ft_epochs * num_batches
    warmup_length = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)

    model.eval()
    if not args.no_regenerate_head:
        with torch.no_grad():
            classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
            classification_head = classification_head.cuda()
    else:
        if args.freeze_encoder:
            classification_head = model
            model = ImageClassifier.load(args.load)
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        else:
            classification_head = model.module.classification_head
    eval_results = evaluate(model, classification_head, args)
    print(f"Pre-finetuning {val_metric_str(args)}:\n {eval_results}")

    val_metrics = {}
    best_val_metric = -float('inf')
    model.train()
    for epoch in range(0, args.ft_epochs):
        epoch_start_time = time.time()
        print(f"Starting epoch {epoch}")
        for i, batch in enumerate(dataloaders["id"]):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            images, labels, text = batch[input_key].cuda(), batch['labels'].cuda(), None
            if args.ft_data is not None and 'metadata' in batch.keys():
                text = batch['metadata'].cuda()
            if args.freeze_encoder:
                logits = model(images)
            else:
                logits, image_features, text_features, logit_scale = model(images, text)
            unlabeled_logits = None
            if dataloaders["unlabeled"] is not None:
                unlabeled_batch = next(iter(dataloaders["unlabeled"]))
                unlabeled_batch = maybe_dictionarize(unlabeled_batch)
                image = unlabeled_batch[input_key].cuda()
                unlabeled_logits, unlabeled_image_features, _, _ = model(image)
                pseudolabels = unlabeled_logits.argmax(dim=-1)
                unlabeled_logits = torch.cat((logits, unlabeled_logits), dim=0)
            if isinstance(loss_fn, LearnedLoss):
                try:
                    ls = logit_scale[0]
                except Exception:
                    ls = logit_scale
                loss = loss_fn(model, logits, labels, image_features, text_features, ls, unlabeled_logits)
            else:
                loss = loss_fn(logits, labels)
                if args.unlabeled_id is not None:
                    loss += loss_fn(unlabeled_logits, pseudolabels)
            loss = loss / args.accumulation_steps
            loss.backward()
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_time = time.time() - start_time
            print_train_update(logger, print_every, total_steps, step, loss, batch_time)

        # Save checkpoints
        model.module.save(os.path.join(args.save, f'checkpoint_{epoch}.pt'))
        print(f"Saved model to {args.save}")

        # Evaluate
        # eval_results = evaluate(model.module, args)
        if not args.no_regenerate_head:
            with torch.no_grad():
                classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
                classification_head = classification_head.cuda()
        else:
            if args.freeze_encoder:
                classification_head = model
                model = ImageClassifier.load(args.load)
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            else:
                classification_head = model.module.classification_head
        eval_results = evaluate(model, classification_head, args)
        val_metrics[epoch] = eval_results
        curr_val_metric = eval_results[val_metric_str(args)]
        if curr_val_metric > best_val_metric:
            print(f"Best val metric of {curr_val_metric} at epoch {epoch}.")
            best_val_metric = curr_val_metric
            model_copy = copy.deepcopy(extract_from_data_parallel(model)).cpu()
            for param in model_copy.parameters():
                param.requires_grad = False
        print(f"Epoch {epoch}, Time: {time.time() - epoch_start_time:.3f}")

    del model
    torch.cuda.empty_cache()
    model_copy = model_copy.cuda()
    model_copy = torch.nn.DataParallel(model_copy, device_ids=list(range(torch.cuda.device_count())))

    return model_copy, best_val_metric


def finetune_final(args, model, loss_fn, optimizer, dataloaders, input_key, print_every, fs_id_dataset=None, fs_val_dataset=None):
    if args.k is not None:
        print(f"Finetuning with {args.k}-shot ID data.")
        # return finetune_fewshot(args, model, loss_fn, optimizer, dataloaders, input_key)
        assert fs_id_dataset is not None and fs_val_dataset is not None, "Please provide few-shot ID and val datasets."
        return finetune_fewshot(args, model, loss_fn, optimizer, dataloaders, fs_id_dataset, fs_val_dataset)
    else:
        print("Finetuning with full ID data.")
        return finetune(args, model, loss_fn, optimizer, dataloaders, input_key, print_every)