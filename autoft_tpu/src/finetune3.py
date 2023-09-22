import json
import os
import time
import re

import src.datasets as datasets
import src.losses as losses
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from src.args import parse_arguments
from src.datasets.common import collate_fn_for_imagenet, collate_fn_for_cifar, FeatureDataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.utils import UnlabeledDatasetWrapper
from src.logger import setup_logging
from src.losses.layerwiseloss import LayerwiseLoss
from src.losses.learnedloss import LearnedLoss
from src.models.eval import _mp_evaluate
from src.models.utils import cosine_lr
from src.models.utils import initialize_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_datasets(args, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    id_dataset = id_dataset_class(preprocess_fn, train=True, n_examples=args.num_id_examples,
                                  location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    all_datasets = {"id": id_dataset}
    return all_datasets


def get_sampler(dataset, train):
    """Helper function to create a sampler."""
    if xm.xrt_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=train)
    else:
        sampler = None
    return sampler


def get_loss_weights(hyperparams, loss_type, num_losses=None):
    if loss_type == "LayerwiseLoss":
        loss_weight_keys = [k for k in hyperparams.keys() if "lossw" in k]
        layer_idx = 0
        layer_loss_weights = []
        loss_weights = []
        for k in loss_weight_keys:
            if int(k.split("_")[0]) == layer_idx:
                layer_loss_weights.append(hyperparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hyperparams[k]]
                layer_idx += 1
    else:
        assert num_losses is not None
        loss_weights = torch.tensor([hyperparams[f"lossw_{i}"] for i in range(num_losses)])
    return loss_weights

def get_dataloader(dataset, is_train, args, sampler=None, image_encoder=None):
    """
    Get a DataLoader for the given dataset.

    Args:
        dataset: Dataset object to be loaded.
        is_train: Boolean indicating if the dataset is for training.
        args: Arguments containing configurations.
        image_encoder: Optional image encoder for feature extraction.

    Returns:
        DataLoader for the given dataset.
    """
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "persistent_workers": args.persistent_workers,
              "prefetch_factor": args.prefetch_factor, "pin_memory": True}
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["sampler"] = get_sampler(dataset, is_train)
    if is_train and kwargs["sampler"] is None:
        kwargs["shuffle"] = True
    else:
        kwargs["shuffle"] = False
    if "ImageNet" in args.id:
        kwargs["collate_fn"] = collate_fn_for_imagenet
    elif "CIFAR" in args.id:
        kwargs["collate_fn"] = collate_fn_for_cifar
    if image_encoder is not None:
        kwargs["collate_fn"] = collate_fn_for_imagenet
        dataset = FeatureDataset(args, is_train, image_encoder, dataset, args.device)
    elif hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    kwargs["dataset"] = dataset
    dataloader = DataLoader(**kwargs)
    device = xm.xla_device()
    dataloader = pl.MpDeviceLoader(dataloader, device, loader_prefetch_size=args.loader_prefetch_size, device_prefetch_size=args.device_prefetch_size)
    return dataloader

def create_optimizer(model, hyperparams, loss_type):
    if loss_type == "LayerwiseLoss":
        layerwise_params = []
        layer_idx = 0
        # Extract layers from the image_encoder (CLIPEncoder) of the model
        for name, module in model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    params_for_layer = {
                        'params': sub_module.parameters(),
                        'lr': hyperparams[f"{layer_idx}_lr"],
                        'weight_decay': hyperparams[f"{layer_idx}_wd"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

        # Classification head of the model
        params_for_layer = {
            'params': model.classification_head.parameters(),
            'lr': hyperparams[f"{layer_idx}_lr"],
            'weight_decay': hyperparams[f"{layer_idx}_wd"]
        }
        layerwise_params.append(params_for_layer)
        optimizer = torch.optim.AdamW(layerwise_params)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
    return optimizer

# def _run2(index, args):
#     device = xm.xla_device()
#     model, preprocess_fn = initialize_model(args)
#     model = model.to(device)
#     _mp_evaluate(index, model, args)

def _run(index, args):
    start_time = time.time()
    device = xm.xla_device()
    model, preprocess_fn = initialize_model(args)
    model = model.to(device)
    if args.eval_only:
        _mp_evaluate(index, model, args)
        return
    if args.method in ["ft-id", "ft-id-ood"]:
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.method in ["autoft"]:
        assert args.load_hparams is not None
        with open(args.load_hparams, 'r') as f:
            best_hparams = json.load(f)
        loss_weights = get_loss_weights(best_hparams, args.loss_type, args.num_losses)
        model_params = [p for p in model.parameters()]
        loss_fn = getattr(losses, args.loss_type)(loss_weights, model_params)
        optimizer = create_optimizer(model, best_hparams, args.loss_type)
        del model_params, loss_weights; torch.cuda.empty_cache()
    all_datasets = get_datasets(args, preprocess_fn)
    train_loader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
    num_batches = len(train_loader) / args.accumulation_steps
    total_steps = num_batches * args.ft_epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)
    per_epoch_eval_results = {}
    if args.freeze_encoder:
        input_key = 'features'
    else:
        input_key = 'images'

    # xm.master_print("Zeroshot accuracies...")
    # _mp_evaluate(index, model, args)
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
    effective_step = 0
    for epoch in range(start_epoch, args.ft_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].to(device)
            labels = batch['labels'].to(device)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                scheduler_step = effective_step + epoch * num_batches
                scheduler(scheduler_step)
                optimizer.zero_grad()
                effective_step += 1
            outputs = model(inputs)
            if isinstance(loss_fn, (LearnedLoss, LayerwiseLoss)):
                loss = loss_fn(outputs, labels, model) / args.accumulation_steps
            else:
                loss = loss_fn(outputs, labels) / args.accumulation_steps
            loss.backward()
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer)

            if i % 100 == 0:
                xm.master_print(f"Epoch [{epoch + 1}/{args.ft_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")
        # Save the current checkpoint along with optimizer and scheduler
        if epoch == args.ft_epochs - 1:
            if xm.is_master_ordinal():
                model_save_path = os.path.join(args.save, f"model_{epoch}.pt")
                model = model.to('cpu')
                model.save(model_save_path)
                model = model.to(device)
                print("Saved model to", model_save_path)
            ckpt_save_path = os.path.join(args.save, f"opt_sched_{epoch}.pt")
            xm.save({
                'optimizer_state': optimizer.state_dict(),
                'scheduler_params': {
                    'base_lrs': args.lr,
                    'warmup_length': args.warmup_length,
                    'steps': total_steps,
                    'current_step': effective_step + epoch * num_batches
                },
                'epoch': epoch
            }, ckpt_save_path)
            xm.master_print(f"Saved optimizer and scheduler to {ckpt_save_path}")
        # Evaluate
        if args.id != "CIFAR10":
            _mp_evaluate(index, model, args)
        xm.master_print(f"Finished training epoch: {epoch + 1} | Time: {time.time() - start_time} s")


def main(args):
    logger = setup_logging(args)
    logger.info(args)
    xmp.spawn(_run, args=(args,), nprocs=8, start_method='spawn')


if __name__ == '__main__':
    args = parse_arguments()
    start_time = time.time()

    os.environ["PYTHONPATH"] = "${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft_tpu/"
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    main(args)
    xm.master_print("Run time: {:.3f} s".format(time.time() - start_time))