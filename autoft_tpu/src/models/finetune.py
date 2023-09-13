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
        return loss_fn(outputs, labels, model)[0]
    return loss_fn(outputs, labels)


def finetune_helper(args, device, model, loss_fn, optimizer, dataloader, input_key, steps=None):
    """Finetune for one epoch or for a specified number of steps."""
    tracker = xm.RateTracker()
    model.train()

    # If steps is not provided, calculate the total number of steps for the entire epoch
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.inner_steps)
    total_steps = len(dataloader) if steps is None else steps

    step = 0
    for batch in dataloader:
        if steps is not None and step >= steps:
            break
        scheduler(step)
        optimizer.zero_grad()

        batch = maybe_dictionarize(batch)
        inputs = batch[input_key]
        labels = batch['labels']

        loss = compute_loss(loss_fn, inputs, labels, model)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(optimizer)
        tracker.add(args.batch_size)
        if step % 100 == 0:
            print_train_update(device, tracker, loss, step, total_steps, epoch=None)
        step += 1
        del batch; del inputs; del labels; del loss

    del tracker; del loss_fn; del optimizer; del scheduler; del dataloader
    gc.collect()

    return model


def _mp_inner_finetune(rank, args, model, loss_fn, optimizer, dataset, input_key):
    torch.manual_seed(args.seed + rank)
    device = xm.xla_device()
    model = model.to(device)
    dataloader = get_dataloader(dataset, is_train=True, args=args)
    model = finetune_helper(args, device, model, loss_fn, optimizer, dataloader, input_key, args.inner_steps)
    return model


def inner_finetune(args, model, loss_fn, optimizer, dataset, input_key):
    """Finetune for inner steps."""
    xmp.spawn(_mp_inner_finetune, args=(args, model, loss_fn, optimizer, dataset, input_key,), nprocs=8, start_method='spawn')


def _mp_finetune(rank, args, model, loss_fn, optimizer, dataset, input_key):
    """Finetune the model on the entire dataset for full epochs."""
    torch.manual_seed(args.seed + rank)
    device = xm.xla_device()
    model = model.to(device)
    dataloader = get_dataloader(dataset, is_train=True, args=args)
    total_steps = len(dataloader) * args.ft_epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)
    per_epoch_eval_results = {}

    # Resume training from the latest checkpoint if it exists
    start_epoch = 0
    ckpts = [f for f in os.listdir(args.save) if re.match(r'checkpoint_\d+\.pt', f)]
    if ckpts:
        last_ckpt = max(ckpts, key=lambda x: int(re.search(r'(\d+)', x).group()))
        ckpt_path = os.path.join(args.save, last_ckpt)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler_params = ckpt['scheduler_params']
        scheduler = cosine_lr(optimizer, scheduler_params['base_lrs'], scheduler_params['warmup_length'],
                              scheduler_params['steps'])
        start_epoch = ckpt['epoch'] + 1  # Start next epoch after the saved epoch
        xm.master_print(f"Found checkpoint {ckpt_path}. Resuming training from epoch {start_epoch}.")
        logger.info(f"Found checkpoint {ckpt_path}. Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.ft_epochs):
        xm.master_print(f"Epoch {epoch} train begin {now()}")
        model = finetune_helper(args, device, model, loss_fn, optimizer, dataloader, input_key, total_steps)
        xm.master_print(f"Epoch {epoch} train end {now()}")
        logger.info(f"Epoch {epoch} train end {now()}")

        # Remove the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_ckpt = os.path.join(args.save, f"checkpoint_{epoch - 1}.pt")
            if xm.master_ordinal() and os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)
                xm.master_print(f"Removed checkpoint {prev_ckpt}")
                logger.info(f"Removed checkpoint {prev_ckpt}")

        # Save the current checkpoint along with optimizer and scheduler
        save_path = os.path.join(args.save, f"checkpoint_{epoch}.pt")
        current_step = epoch * len(dataloader)
        xm.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_params': {
                'base_lrs': args.lr,
                'warmup_length': args.warmup_length,
                'steps': total_steps,
                'current_step': current_step
            },
            'epoch': epoch
        }, save_path)
        xm.master_print(f"Saved model, optimizer, and scheduler to {save_path}")
        logger.info(f"Saved model, optimizer, and scheduler to {save_path}")

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