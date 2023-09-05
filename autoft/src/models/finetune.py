import json
import logging
import os
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from src.datasets.common import get_dataloader
from src.datasets.common import maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr, get_device, is_tpu_available
from src.models.sampler import get_sampler

logger = logging.getLogger('main')

def print_train_update(logger, print_every, total_steps, step, loss, batch_time, data_time):
    if is_tpu_available():
        should_print = (print_every is not None and step % print_every == 0) if xm.is_master_ordinal() else False
    else:
        should_print = (print_every is not None and step % print_every == 0)
    if should_print:
        percent_complete = 100 * step / total_steps
        xm.master_print(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
            f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)
        logger.info(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}")

def test_finetuned_model(args, model, all_eval_results, total_steps):
    """Evaluate the model on the test set and save the results."""
    should_eval = xm.is_master_ordinal() if is_tpu_available() else True
    if should_eval:
        args.current_epoch = args.ft_epochs
        eval_results = evaluate(model.module, args)
        xm.master_print(eval_results)
        logger.info(json.dumps(eval_results, indent=4))
        all_eval_results[total_steps] = eval_results
        os.makedirs(args.save, exist_ok=True)
        results_path = os.path.join(args.save, 'eval_results.json')
        with open(results_path, 'wb') as f:
            f.write(json.dumps(all_eval_results) + '\n')
        xm.master_print(f'\nSaved evaluation results to {results_path}.')

def save_finetuned_model(args, model, optimizer, logger):
    should_save = xm.is_master_ordinal() if args.use_tpu else True
    if should_save:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.ft_epochs}.pt')
        xm.master_print('Saving model to', model_path)
        logger.info(f"Saving model to {model_path}")
        model.module.save(model_path)
        optim_path = os.path.join(args.save, f'optim_{args.ft_epochs}.pt')
        torch.save(optimizer.state_dict(), optim_path)
        xm.master_print('Saving optimizer to', optim_path)
        logger.info(f"Saving optimizer to {optim_path}")
        return model_path

def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every):
    device = get_device()
    if not is_tpu_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()

    params = list(model.parameters())
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.inner_steps)
    if is_tpu_available():
        dataloader = pl.MpDeviceLoader(dataloader, device)

    for step in range(args.inner_steps):
        batch = next(iter(dataloader))
        scheduler(step)
        optimizer.zero_grad()

        start_time = time.time()
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].to(device)
        labels = batch['labels'].to(device)
        data_time = time.time() - start_time

        if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
            loss, _ = loss_fn(inputs, labels, model)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if is_tpu_available():
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
        batch_time = time.time() - start_time

        print_train_update(logger, print_every, args.inner_steps, step, loss, batch_time, data_time)

    return model

def finetune_final(args, model, loss_fn, optimizer, dataset, input_key, print_every, datapoint_weights=None):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    device = get_device()
    if not is_tpu_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()
    params = list(model.parameters())
    sampler = get_sampler(dataset, datapoint_weights, shuffle=True)
    dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=sampler)
    if is_tpu_available():
        dataloader = pl.MpDeviceLoader(dataloader, device)
    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)
    all_eval_results = {}

    for epoch in range(args.ft_epochs):
        model.train()

        for i, batch in enumerate(dataloader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            start_time = time.time()
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].to(device)
            labels = batch['labels'].to(device)
            data_time = time.time() - start_time

            if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
                loss, _ = loss_fn(inputs, labels, model)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if args.use_tpu:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            batch_time = time.time() - start_time

            print_train_update(logger, print_every, args.inner_steps, step, loss, batch_time, data_time)
            # eval_results = evaluate(model.module, args)
            # all_eval_results[step] = eval_results

    # Test finetuned model
    test_finetuned_model(args, model, all_eval_results, total_steps)

    # Save finetuned model
    save_finetuned_model(args, model, optimizer, logger)

    return model.module