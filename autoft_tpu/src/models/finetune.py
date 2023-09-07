import logging
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import evaluate
from src.models.utils import cosine_lr, get_device, is_tpu_available, print_train_update
from src.models.sampler import get_sampler

logger = logging.getLogger('main')


def get_initialized_model(model, device):
    """Returns model initialized with device and DataParallel if necessary."""
    if not is_tpu_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model.to(device)


def compute_loss(loss_fn, inputs, labels, model):
    """Computes the loss using either LearnedLoss, LayerwiseLoss, or default method."""
    if isinstance(loss_fn, (LearnedLoss, LayerwiseLoss)):
        return loss_fn(inputs, labels, model)[0]
    outputs = model(inputs)
    return loss_fn(outputs, labels)


def train_step(batch, input_key, device, loss_fn, model, optimizer, scheduler, step):
    """Executes a single training step and returns the loss."""
    scheduler(step)
    optimizer.zero_grad()

    start_time = time.time()
    batch = maybe_dictionarize(batch)
    inputs = batch[input_key].to(device)
    labels = batch['labels'].to(device)
    data_time = time.time() - start_time

    loss = compute_loss(loss_fn, inputs, labels, model)
    loss.backward()

    params = list(model.parameters())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    if is_tpu_available():
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()

    return loss, time.time() - start_time, data_time


def finetune(args, model, loss_fn, optimizer, dataloader, input_key, steps, print_every, is_inner=True):
    """Finetunes the model for the given steps."""
    device = get_device()
    model = get_initialized_model(model, device)
    model.train()

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, steps)
    if is_tpu_available():
        dataloader = pl.MpDeviceLoader(dataloader, device)

    for step in range(steps):
        batch = next(iter(dataloader))
        loss, batch_time, data_time = train_step(batch, input_key, device, loss_fn, model, optimizer, scheduler, step)
        print_train_update(logger, print_every, steps, step, loss, batch_time, data_time)

    return model if is_inner else model.module


def inner_finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every):
    """Finetune for inner steps."""
    return finetune(args, model, loss_fn, optimizer, dataloader, input_key, args.inner_steps, print_every, True)


def finetune_final(args, model, loss_fn, optimizer, dataset, input_key, print_every):
    """Finetune the model on the entire dataset for full epochs."""
    assert args.load is not None, "Please provide the path to a checkpoint through --load."
    device = get_device()
    model = get_initialized_model(model, device)

    sampler = get_sampler(dataset, shuffle=True)
    dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None, sampler=sampler)

    num_batches = len(dataloader)
    total_steps = args.ft_epochs * num_batches

    model = finetune(args, model, loss_fn, optimizer, dataloader, input_key, total_steps, print_every, False)

    if args.plot:
        eval_results = evaluate(model, args)
        return model, eval_results

    return model
