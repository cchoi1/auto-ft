import json
import logging
import os
import re

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr, print_train_update, now
import torch_xla.debug.metrics as met

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

    for step, batch in enumerate(dataloader):
        if steps is not None and step >= steps:
            break

        scheduler(step)
        optimizer.zero_grad()

        batch = maybe_dictionarize(batch)
        inputs = batch[input_key]
        labels = batch['labels']

        loss = compute_loss(loss_fn, inputs, labels, model)
        loss.backward()

        params = list(model.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        xm.optimizer_step(optimizer)
        tracker.add(args.batch_size)
        if step % 100 == 0:
            print_train_update(device, tracker, loss, step, total_steps, epoch=None)

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
    checkpoints = [f for f in os.listdir(args.save) if re.match(r'checkpoint_\d+\.pt', f)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'(\d+)', x).group()))
        checkpoint_path = os.path.join(args.save, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1  # Start next epoch after the saved epoch
        xm.master_print(f"Found checkpoint {checkpoint_path}. Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.ft_epochs):
        xm.master_print(f"Epoch {epoch} train begin {now()}")
        model = finetune_helper(args, device, model, loss_fn, optimizer, dataloader, input_key, total_steps)
        xm.master_print(f"Epoch {epoch} train end {now()}")

        # Remove the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint = os.path.join(args.save, f"checkpoint_{epoch - 1}.pt")
            if xm.master_ordinal() and os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)
                xm.master_print(f"Removed checkpoint {prev_checkpoint}")

        # Save the current checkpoint along with optimizer and scheduler
        save_path = os.path.join(args.save, f"checkpoint_{epoch}.pt")
        xm.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch
        }, save_path)
        xm.master_print(f"Saved model, optimizer, and scheduler to {save_path}")

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
        rank = xm.get_ordinal()
        ft_results = _mp_finetune(rank, args, model, loss_fn, optimizer, dataset, input_key)
        return ft_results