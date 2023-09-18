import json
import logging
import os
import re
import gc

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr, print_train_update, now

logger = logging.getLogger('main')

def compute_loss(loss_fn, inputs, labels, model):
    """Computes the loss using either LearnedLoss, LayerwiseLoss, or default method."""
    outputs = model(inputs)
    if isinstance(loss_fn, (LearnedLoss, LayerwiseLoss)):
        return loss_fn(outputs, labels, model)
    return loss_fn(outputs, labels)


def finetune_helper(args, model, loss_fn, optimizer, dataloader, input_key, steps, accumulation_steps=1):
    """Finetune for one epoch or for a specified number of steps."""
    device = xm.xla_device()
    model = model.to(device)
    model.train()
    effective_step = 0

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key]
        labels = batch['labels']

        if (step + 1) % accumulation_steps == 0:
            effective_step += 1
            optimizer.zero_grad()
        loss = compute_loss(loss_fn, inputs, labels, model) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            xm.mark_step()

        # Delete unnecessary variables
        # del batch, inputs, labels, loss
        # gc.collect()
        # Empty the cache
        # torch.cuda.empty_cache()

    # del loss_fn, optimizer, dataloader
    # gc.collect()

    return model



def _mp_inner_finetune(rank, args, model, loss_fn, optimizer, scheduler, dataset, input_key):
    torch.manual_seed(args.seed + rank)
    device = xm.xla_device()
    model = model.to(device)
    dataloader = get_dataloader(dataset, is_train=True, args=args)
    model = finetune_helper(args, model, loss_fn, optimizer, dataloader, input_key, args.inner_steps, args.accumulation_steps)
    return model


def inner_finetune(args, model, loss_fn, optimizer, scheduler, dataset, input_key):
    """Finetune for inner steps."""
    xmp.spawn(_mp_inner_finetune, args=(args, model, loss_fn, optimizer, scheduler, dataset, input_key,), nprocs=8, start_method='spawn')


def has_parameters_changed(model, old_params):
    for param, old_param in zip(model.parameters(), old_params):
        if not torch.equal(param.data, old_param.data):
            return True
    return False

def _mp_finetune(rank, args, model, loss_fn, optimizer, dataset, input_key):
    """Finetune the model on the entire dataset for full epochs."""
    device = xm.xla_device()
    model = model.to(device)
    model.train()
    tracker = xm.RateTracker()

    dataloader = get_dataloader(dataset, is_train=True, args=args)
    num_batches = len(dataloader)
    total_steps = int(num_batches / args.accumulation_steps) * args.ft_epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)
    per_epoch_eval_results = {}

    # Resume training from the latest checkpoint if it exists
    start_epoch = 0
    model_epochs = [int(re.search(r'model_(\d+)\.pt', f).group(1)) for f in os.listdir(args.save) if
                    re.match(r'model_\d+\.pt', f)]
    opt_sched_epochs = [int(re.search(r'opt_sched_(\d+)\.pt', f).group(1)) for f in os.listdir(args.save) if
                        re.match(r'opt_sched_\d+\.pt', f)]
    if len(model_epochs) > 0 and len(opt_sched_epochs) > 0:
        last_epoch = min(max(model_epochs), max(opt_sched_epochs))
        model_ckpt_path = os.path.join(args.save, f'model_{last_epoch}.pt')
        opt_sched_ckpt_path = os.path.join(args.save, f'opt_sched_{last_epoch}.pt')
        model.load(model_ckpt_path)
        xm.master_print(f"Loaded model checkpoint from epoch {last_epoch}.")
        opt_sched_ckpt = torch.load(opt_sched_ckpt_path)
        optimizer.load_state_dict(opt_sched_ckpt['optimizer_state'])
        scheduler_params = opt_sched_ckpt['scheduler_params']
        scheduler = cosine_lr(optimizer, scheduler_params['base_lrs'], scheduler_params['warmup_length'],
                              scheduler_params['steps'])
        start_epoch = last_epoch + 1  # Start next epoch after the saved epoch
        xm.master_print(f"Found checkpoint for epoch {last_epoch}. Resuming training from epoch {start_epoch}.")
        logger.info(f"Found checkpoint for epoch {last_epoch}. Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.ft_epochs):
        i = 0
        effective_step = 0
        xm.master_print(f"Epoch {epoch} train begin {now()}")
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].to(device)
            labels = batch['labels'].to(device)
            if (i + 1) % args.accumulation_steps == 0:
                scheduler_step = effective_step + epoch * num_batches
                scheduler(scheduler_step)
                optimizer.zero_grad()
                effective_step += 1
            # loss = compute_loss(loss_fn, inputs, labels, model) / args.accumulation_steps
            outputs = model(inputs)
            if isinstance(loss_fn, (LearnedLoss, LayerwiseLoss)):
                loss = loss_fn(outputs, labels, model) / args.accumulation_steps
            else:
                loss = loss_fn(outputs, labels) / args.accumulation_steps
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer, barrier=True)
                tracker.add(args.accumulation_steps * args.batch_size)
            i += 1
            del batch, inputs, labels, loss
        xm.master_print(f"Epoch {epoch} train end {now()}")
        logger.info(f"Epoch {epoch} train end {now()}")

        # Remove the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_ckpt = os.path.join(args.save, f"checkpoint_{epoch - 1}.pt")
            if xm.get_ordinal() == 0 and os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)
                xm.master_print(f"Removed checkpoint {prev_ckpt}")
                logger.info(f"Removed checkpoint {prev_ckpt}")

        # Save the current checkpoint along with optimizer and scheduler
        model_save_path = os.path.join(args.save, f"model_{epoch}.pt")
        model.save(model_save_path)
        ckpt_save_path = os.path.join(args.save, f"opt_sched_{epoch}.pt")
        xm.save({
            'optimizer_state': optimizer.state_dict(),
            'scheduler_params': {
                'base_lrs': args.lr,
                'warmup_length': args.warmup_length,
                # 'steps': total_steps,
                'current_step': effective_step + epoch * num_batches
            },
            'epoch': epoch
        }, ckpt_save_path)
        xm.master_print(f"Saved optimizer and scheduler to {ckpt_save_path}")
        logger.info(f"Saved optimizer and scheduler to {ckpt_save_path}")
        gc.collect()

        if args.plot or epoch == args.ft_epochs - 1:
            xm.rendezvous('update_barrier')
            epoch_eval_results = evaluate(model, args, spawn_required=False)
            xm.master_print(epoch_eval_results)
            logger.info(json.dumps(epoch_eval_results, indent=4))
            per_epoch_eval_results[epoch] = epoch_eval_results

    ft_results = {"model": model, "eval_results": per_epoch_eval_results}
    return ft_results



def finetune(args, model, loss_fn, optimizer, dataset, input_key, spawn_required=True):
    if spawn_required:
        xmp.spawn(_mp_finetune, args=(args,model,loss_fn,optimizer,dataset,input_key,), nprocs=8, start_method='spawn')
    else:
        rank = xm.xla_device()
        ft_results = _mp_finetune(rank, args, model, loss_fn, optimizer, dataset, input_key)
        return ft_results