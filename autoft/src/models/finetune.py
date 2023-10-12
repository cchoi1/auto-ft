import logging
import os
import time
import numpy as np

import torch

from src.datasets.common import maybe_dictionarize
from src.losses.learnedloss import LearnedLoss
from src.models import utils
from src.models.modeling import ImageClassifier
from src.models.eval import evaluate
from src.models.utils import cosine_lr, extract_from_data_parallel

logger = logging.getLogger('main')

def print_train_update(logger, print_every, total_steps, step, loss, batch_time, data_time):
    should_print = (print_every is not None and step % print_every == 0)
    if should_print:
        percent_complete = 100 * step / total_steps
        print(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
            f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)
        logger.info(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}")

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


# def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, unlabeled_dataloader=None):
#     inner_finetune_start_time = time.time()
#     warmup_steps = args.warmup_length * args.accumulation_steps
#     scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.inner_steps)
#
#     model.train()
#     unlabeled_logits, pseudolabels = None, None
#
#     num_steps = args.inner_steps * args.accumulation_steps
#     step = 0
#     while step < num_steps:
#         for batch in dataloader:
#             if step >= num_steps:
#                 break
#             scheduler(step)
#             if step % args.accumulation_steps == 0:
#                 optimizer.zero_grad()
#
#             batch = maybe_dictionarize(batch)
#             inputs = batch[input_key].cuda()
#             labels = batch['labels'].cuda()
#             if unlabeled_dataloader is not None:
#                 unlabeled_batch = next(iter(unlabeled_dataloader))
#                 unlabeled_batch = maybe_dictionarize(unlabeled_batch)
#                 pseudolabels = unlabeled_logits.argmax(dim=-1)
#             if args.ft_data is not None:
#                 image, text = inputs, batch['metadata']
#                 image_features, text_features, logit_scale = model(image, text)
#             # logits, image_features, text_features, logit_scale = model(image, text)
#
#             if isinstance(loss_fn, LearnedLoss):
#                 loss = loss_fn(model, logits, labels, image_features, text_features, logit_scale)
#             else:
#                 loss = loss_fn(logits, labels)
#                 if args.unlabeled_id is not None:
#                     loss += loss_fn(unlabeled_logits, pseudolabels)
#             loss = loss / args.accumulation_steps
#             loss.backward()
#             if (step + 1) % args.accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
#                 optimizer.step()
#             step += 1
#
#     torch.cuda.empty_cache()
#     print(f"Time to finetune: {time.time() - inner_finetune_start_time:.3f}", flush=True)
#
#     return model


def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, unlabeled_dataloader=None, image_encoder=None):
    torch.cuda.synchronize()
    fn_start = time.time()
    model.train()
    warmup_steps = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.inner_steps)
    num_steps = args.inner_steps * args.accumulation_steps
    unlabeled_logits, pseudolabels = None, None
    image_features, text_features, logit_scale = None, None, None

    batch_prep_times, inner_step_times = [], []
    loss_times, backprop_times = [], []
    print("num batches", len(dataloader))
    for step, batch in enumerate(dataloader):
        torch.cuda.synchronize()
        start = time.time()
        if step >= num_steps:
            break
        scheduler(step)

        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        labels = batch['labels'].cuda()
        torch.cuda.synchronize()
        batch_prep_times.append(time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        # logits = utils.get_logits(inputs, model)
        if unlabeled_dataloader is not None:
            unlabeled_batch = next(iter(unlabeled_dataloader))
            unlabeled_batch = maybe_dictionarize(unlabeled_batch)
            unlabeled_logits = utils.get_logits(unlabeled_batch[input_key].cuda(), model)
            pseudolabels = unlabeled_logits.argmax(dim=-1)
        if args.ft_data is not None:
            image, text = inputs, batch['metadata']
            # image_features, text_features, logit_scale = image_encoder(image, text)
        logits, image_features, text_features, logit_scale = model(image, text)
        torch.cuda.synchronize()
        inner_step_times.append(time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        if isinstance(loss_fn, LearnedLoss):
            # loss = loss_fn(logits, labels, model, unlabeled_logits, pseudolabels, image_features, text_features, logit_scale)
            loss = loss_fn(model, logits, labels, image_features, text_features, logit_scale)
        else:
            loss = loss_fn(logits, labels)
            if args.unlabeled_id is not None:
                loss += loss_fn(unlabeled_logits, pseudolabels)
        loss = loss / args.accumulation_steps
        torch.cuda.synchronize()
        loss_times.append(time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        backprop_times.append(time.time() - start)
    print(f"    Time per batch prep: {np.mean(batch_prep_times):.3f} x {len(batch_prep_times)} = {sum(batch_prep_times):.3f}", flush=True)
    print(f"    Time per inner step: {np.mean(inner_step_times):.3f} x {len(inner_step_times)} = {sum(inner_step_times):.3f}", flush=True)
    print(f"    Time per loss calc: {np.mean(loss_times):.3f} x {len(loss_times)} = {sum(loss_times):.3f}", flush=True)
    print(f"    Time per backprop: {np.mean(backprop_times):.3f} x {len(backprop_times)} = {sum(backprop_times):.3f}", flush=True)

    start = time.time()
    torch.cuda.empty_cache()
    print(f"    Time to empty cache: {time.time() - start:.3f}", flush=True)
    print(f"    Total time before return: {time.time() - fn_start:.3f}", flush=True)
    return model


def find_latest_checkpoint(save_dir):
    """Returns the latest checkpoint file and its epoch number."""
    checkpoints = [filename for filename in os.listdir(save_dir) if filename.startswith('checkpoint_')]
    if not checkpoints:
        return None, -1
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.pt')[0]))
    latest_checkpoint = checkpoints[-1]
    epoch_number = int(latest_checkpoint.split('_')[1].split('.pt')[0])
    return os.path.join(save_dir, latest_checkpoint), epoch_number


def finetune_final(args, model, loss_fn, optimizer, dataloader, input_key, print_every, unlabeled_dataloader=None, image_encoder=None):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches
    warmup_length = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)
    unlabeled_logits, pseudolabels = None, None
    image_features, text_features, logit_scale = None, None, None
    best_val_metric = -float('inf')
    patience = 0
    all_eval_results = {}
    model.train()

    # Find the latest checkpoint and resume from there
    start_epoch = 0
    checkpoint_path, latest_epoch = find_latest_checkpoint(args.save)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = latest_epoch + 1
        print(f"Resuming from epoch {start_epoch} using checkpoint {checkpoint_path}")

    print("num batches", len(dataloader))
    for epoch in range(start_epoch, args.ft_epochs):
        print(f"Starting epoch {epoch}...", flush=True)
        epoch_start_time = time.time()

        for i, batch in enumerate(dataloader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            # logits = utils.get_logits(inputs, model)
            if unlabeled_dataloader is not None:
                unlabeled_batch = next(iter(unlabeled_dataloader))
                unlabeled_batch = maybe_dictionarize(unlabeled_batch)
                # unlabeled_logits = utils.get_logits(unlabeled_batch[input_key].cuda(), model)
                pseudolabels = unlabeled_logits.argmax(dim=-1)

            if args.ft_data is not None:
                image, text = inputs, batch['metadata']
                # image_features, text_features, logit_scale = image_encoder(image, text)
            logits, image_features, text_features, logit_scale = model(image, text)

            if isinstance(loss_fn, LearnedLoss):
                loss = loss_fn(model, logits, labels, image_features, text_features, logit_scale)
                # loss = loss_fn(logits, labels, model, unlabeled_logits, pseudolabels, image_features, text_features,
                #                logit_scale)
            else:
                # logits = utils.get_logits(inputs, model)
                loss = loss_fn(logits, labels)
                if args.unlabeled_id is not None:
                    loss += loss_fn(unlabeled_logits, pseudolabels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_time = time.time() - start_time
            print_train_update(logger, print_every, total_steps, step, loss, batch_time, data_time)

        # Save the checkpoint after each epoch
        # checkpoint = {
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict()
        # }
        # torch.save(checkpoint, os.path.join(args.save, f'opt_sched_checkpoint_{epoch}.pt'))
        # print(f"Saved optimizer and scheduler to {args.save}/opt_sched_checkpoint_{epoch}.pt", flush=True)

        eval_results = evaluate(model.module, args)
        # eval_results = evaluate(encoder, classification_head, args)
        all_eval_results[step] = eval_results
        print(f"Epoch time: {time.time() - epoch_start_time:.3f}", flush=True)
        temp_model = extract_from_data_parallel(model)
        # temp_model = ImageClassifier(encoder, classification_head)
        temp_model.save(os.path.join(args.save, f'checkpoint_{epoch}.pt'))
        print(f"Saved model to {args.save}", flush=True)
        del temp_model; torch.cuda.empty_cache()

        current_val_metric = eval_results['acc']
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            patience = 0
        else:
            patience += 1
        if patience > args.early_stopping_patience:
            print("Early stopping.")
            break

    del dataloader; torch.cuda.empty_cache()

    return model
