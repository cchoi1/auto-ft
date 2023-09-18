import json
import logging
import os
import time

import torch
from src.datasets.common import get_dataloader
from src.datasets.common import maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr, get_device

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


def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every, id_val_acc_thresh=None, id_val_dataloader=None):
    model.train()

    params = list(model.parameters())
    warmup_steps = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.inner_steps)
    num_steps = args.inner_steps * args.accumulation_steps

    for step in range(num_steps):
        batch = next(iter(dataloader))
        scheduler(step)
        if step % args.accumulation_steps == 0:
            optimizer.zero_grad()

        start_time = time.time()
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        labels = batch['labels'].cuda()
        data_time = time.time() - start_time

        if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
            loss = loss_fn(inputs, labels, model)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        loss = loss / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        batch_time = time.time() - start_time
        print_train_update(logger, print_every, args.inner_steps, step, loss, batch_time, data_time)

        # Check accuracy for early stopping
        if (step + 1) % args.accumulation_steps == 0 and id_val_acc_thresh is not None:
            acc = compute_accuracy(model, id_val_dataloader, input_key)
            if acc >= id_val_acc_thresh:
                print(f"Early stopping at step {step} with accuracy {acc:.2f}")
                break

    torch.cuda.empty_cache()

    return model


def finetune_final(args, model, loss_fn, optimizer, dataset, input_key, print_every):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."

    dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=None)
    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches
    warmup_length = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)
    all_eval_results = {}

    for epoch in range(args.ft_epochs):
        print(f"Starting epoch {epoch}...", flush=True)
        epoch_start_time = time.time()
        model.train()
        dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=None)
        print("Got dataloader")

        for i, batch in enumerate(dataloader):
            step = i + epoch * num_batches
            scheduler(step)

            start_time = time.time()
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
                loss = loss_fn(inputs, labels, model)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                # del outputs  # free up some memory
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_time = time.time() - start_time
            print_train_update(logger, print_every, total_steps, step, loss, batch_time, data_time)
            # del inputs, labels, loss

        if args.plot and args.id != "ImageNet":
            with torch.no_grad():  # Evaluation doesn't require gradient computation
                eval_results = evaluate(model.module, args)
            all_eval_results[step] = eval_results
        print(f"Epoch time: {time.time() - epoch_start_time:.3f}", flush=True)

    del dataloader, dataset
    torch.cuda.empty_cache()

    # save_finetuned_model(args, model, optimizer, logger)

    return model
