import json
import logging
import os
import time

import torch
from src.datasets.common import maybe_dictionarize
from src.losses.learnedloss import LearnedLoss
from src.losses.layerwiseloss import LayerwiseLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr
from torch.nn.parallel import DistributedDataParallel as DDP
from src.datasets.common import get_dataloader

logger = logging.getLogger('main')

def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model.to(local_rank), device_ids=[local_rank])

    model.train()
    params = list(model.parameters())
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.inner_steps)

    for step in range(args.inner_steps):
        batch = next(iter(dataloader))
        scheduler(step)
        optimizer.zero_grad()

        start_time = time.time()
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        labels = batch['labels'].cuda()
        data_time = time.time() - start_time

        if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
            loss, _ = loss_fn(inputs, labels, model)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        batch_time = time.time() - start_time

        if print_every is not None and step % print_every == 0:
            percent_complete = 100 * step / args.inner_steps
            print(f"Train Iter: {step}/{args.inner_steps} [{percent_complete:.0f}% ]\t"
                  f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)
            logger.info(f"Train Iter: {step}/{args.inner_steps} [{percent_complete:.0f}% ]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}")

    return model.module

def finetune_final(args, model, loss_fn, optimizer, dataset, input_key, print_every, sampler=None):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model.to(local_rank), device_ids=[local_rank])

    model.train()
    params = list(model.parameters())
    dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=sampler)
    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)
    start_steps_time = time.time()
    all_eval_results = {}

    for epoch in range(args.ft_epochs):
        model.train()
        dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=sampler)

        for i, batch in enumerate(dataloader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            start_time = time.time()
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            if isinstance(loss_fn, LearnedLoss) or isinstance(loss_fn, LayerwiseLoss):
                loss, _ = loss_fn(inputs, labels, model)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if print_every is not None and step % print_every == 0:
                steps_time = time.time() - start_steps_time
                percent_complete = 100 * step / total_steps
                print(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                      f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                      f"Steps (t) {steps_time:.3f}", flush=True)
                logger.info(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                            f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}")
                eval_results = evaluate(model.module, args)
                all_eval_results[step] = eval_results
                start_steps_time = time.time()

    # Final evaluation and save
    args.current_epoch = args.ft_epochs
    eval_results = evaluate(model.module, args)
    print(eval_results)
    logger.info(json.dumps(eval_results, indent=4))
    all_eval_results[total_steps] = eval_results
    results_path = os.path.join(args.save, 'eval_results.json')
    with open(results_path, 'a+') as f:
        f.write(json.dumps(all_eval_results) + '\n')
    print(f'\nSaved evaluation results to {results_path}.')

    # Save finetuned checkpoint
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.ft_epochs}.pt')
        print('Saving model to', model_path)
        logger.info(f"Saving model to {model_path}")
        model.module.save(model_path)
        optim_path = os.path.join(args.save, f'optim_{args.ft_epochs}.pt')
        torch.save(optimizer.state_dict(), optim_path)
        return model_path

    return model.module