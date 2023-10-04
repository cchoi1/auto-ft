import logging
import os
import time

import torch

from src.datasets.common import maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models import utils
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


def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, unlabeled_dataloader=None, image_encoder=None):
    inner_finetune_start_time = time.time()
    model.train()
    warmup_steps = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.inner_steps)
    num_steps = args.inner_steps * args.accumulation_steps
    step = 0
    unlabeled_logits, pseudolabels = None, None
    image_features, text_features, logit_scale = None, None, None

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            scheduler(step)
            if step % args.accumulation_steps == 0:
                optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            logits = utils.get_logits(inputs, model)
            if unlabeled_dataloader is not None:
                unlabeled_batch = next(iter(unlabeled_dataloader))
                unlabeled_batch = maybe_dictionarize(unlabeled_batch)
                unlabeled_logits = utils.get_logits(unlabeled_batch[input_key].cuda(), model)
                pseudolabels = unlabeled_logits.argmax(dim=-1)
            if args.ft_data is not None:
                image, text = inputs, batch['metadata']
                image_features, text_features, logit_scale = image_encoder(image, text)

            if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
                loss = loss_fn(logits, labels, model, unlabeled_logits, pseudolabels, image_features, text_features, logit_scale)
            else:
                loss = loss_fn(logits, labels)
                if args.unlabeled_id is not None:
                    loss += loss_fn(unlabeled_logits, pseudolabels)
            loss = loss / args.accumulation_steps
            loss.backward()
            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                optimizer.step()
            step += 1

    torch.cuda.empty_cache()
    print(f"Time to finetune: {time.time() - inner_finetune_start_time:.3f}", flush=True)

    return model


def finetune_final(args, model, loss_fn, optimizer, dataloader, input_key, print_every, unlabeled_dataloader=None, image_encoder=None):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches
    warmup_length = args.warmup_length * args.accumulation_steps
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)
    all_eval_results = {}
    model.train()
    unlabeled_logits, pseudolabels = None, None
    image_features, text_features, logit_scale = None, None, None

    for epoch in range(args.ft_epochs):
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
            logits = utils.get_logits(inputs, model)
            if unlabeled_dataloader is not None:
                unlabeled_batch = next(iter(unlabeled_dataloader))
                unlabeled_batch = maybe_dictionarize(unlabeled_batch)
                unlabeled_logits = utils.get_logits(unlabeled_batch[input_key].cuda(), model)
                pseudolabels = unlabeled_logits.argmax(dim=-1)
            if args.ft_data is not None:
                image, text = inputs, batch['metadata']
                image_features, text_features, logit_scale = image_encoder(image, text)

            if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
                loss = loss_fn(logits, labels, model, unlabeled_logits, pseudolabels, image_features, text_features, logit_scale)
            else:
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

        if args.id != "ImageNet":
            eval_results = evaluate(model.module, args)
            all_eval_results[step] = eval_results
        print(f"Epoch time: {time.time() - epoch_start_time:.3f}", flush=True)
        temp_model = extract_from_data_parallel(model)
        temp_model.save(os.path.join(args.save, f'checkpoint_{epoch}.pt'))
        print(f"Saved model to {args.save}", flush=True)
        del temp_model; torch.cuda.empty_cache()

    del dataloader
    torch.cuda.empty_cache()
    save_finetuned_model(args, model, optimizer, logger)

    return model
