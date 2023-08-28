import json
import logging
import os
import time

import torch
from src.datasets.common import maybe_dictionarize
from src.losses.layerloss import LayerLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('main')

def finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    if args.distributed:
        rank = torch.distributed.get_rank()
        model = DDP(model.to(rank), device_ids=[rank])

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

        if isinstance(loss_fn, LayerLoss):
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

    # Evaluate
    if args.method != "autoft":
        args.current_epoch = args.inner_steps
        eval_results = evaluate(model.module, args)
        print(eval_results)
        logger.info(json.dumps(eval_results, indent=4))

    if args.save is not None and args.method != "autoft":
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.inner_steps}.pt')
        print('Saving model to', model_path)
        logger.info(f"Saving model to {model_path}")
        model.module.save(model_path)
        optim_path = os.path.join(args.save, f'optim_{args.inner_steps}.pt')
        torch.save(optimizer.state_dict(), optim_path)
        if args.save is not None:
            return model_path

    return model.module