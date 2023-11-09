import json
import logging
import os
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import src.datasets as datasets
from src.args import parse_arguments
from src.datasets.common import collate_fn_for_imagenet, collate_fn_for_cifar, FeatureDataset
from src.datasets.common import maybe_dictionarize
from src.logger import setup_logging
from src.losses.learnedloss import LearnedLoss
from src.models.eval import _mp_evaluate
from src.models.modeling import ImageClassifier
from src.models.utils import cosine_lr
from src.models.utils import set_seed, print_hparams, save_hparams
from src.datasets.laion import get_data


def initialize_model(args):
    image_classifier = ImageClassifier.load(args.load)
    if args.freeze_encoder:
        model = image_classifier.classification_head
        preprocess_fn = image_classifier.val_preprocess
    else:
        model = image_classifier
        preprocess_fn = image_classifier.train_preprocess
        image_classifier.process_images = True

    return model, preprocess_fn


def get_datasets(args, model, preprocess_fn):
    id_dataset_class = getattr(datasets, args.id)
    if args.ft_data is not None:
        train_preprocess_fn = model.image_encoder.train_preprocess
        val_preprocess_fn = model.image_encoder.val_preprocess
        id_dataset = get_data(args, (train_preprocess_fn, val_preprocess_fn), epoch=0)
    else:
        id_dataset = id_dataset_class(preprocess=preprocess_fn, train=True, n_examples=args.num_id_examples,
                                      location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    all_datasets = {"id": id_dataset}
    return all_datasets


def get_sampler(dataset, train):
    """Helper function to create a sampler."""
    if xm.xrt_world_size() > 1:
        print(f"Using distributed sampler for with {xm.xrt_world_size()} replicas")
        sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=train)
    else:
        sampler = None
    return sampler


def get_loss_weights(hparams, layerwise):
    global_loss_weight_keys = [k for k in sorted(hparams.keys()) if "lossw" in k and "_lossw" not in k]
    global_loss_weights = torch.tensor([hparams[k] for k in global_loss_weight_keys])
    if layerwise:
        layerwise_loss_weight_keys = [k for k in hparams.keys() if "_lossw" in k]
        layerwise_loss_weight_keys = sorted(layerwise_loss_weight_keys,
                                            key=lambda x: (int(x.split("_")[0]), x.split("_")[2]))
        layer_idx = 0
        layer_loss_weights = []
        loss_weights = []
        for k in layerwise_loss_weight_keys:
            if int(k.split("_")[0]) == layer_idx:
                layer_loss_weights.append(hparams[k])
            else:
                loss_weights.append(torch.tensor(layer_loss_weights))
                layer_loss_weights = [hparams[k]]
                layer_idx += 1
        layerwise_loss_weights = torch.stack(loss_weights)
        global_loss_weights = global_loss_weights.expand(layerwise_loss_weights.shape[0], -1)
        loss_weights = torch.cat([global_loss_weights, layerwise_loss_weights], dim=1)
    else:
        loss_weights = global_loss_weights
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
              "prefetch_factor": args.prefetch_factor}
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
    print('hi finetune3 get_dataloader')
    print(f"Num batches: {len(dataloader)}")
    dataloader = pl.MpDeviceLoader(dataloader, device, loader_prefetch_size=args.loader_prefetch_size, device_prefetch_size=args.device_prefetch_size)
    return dataloader

def create_optimizer(model, hparams, layerwise=False):
    if layerwise:
        layerwise_params = []
        layer_idx = 0
        # Extract layers from the image_encoder (CLIPEncoder) of the model
        for name, module in model.image_encoder.named_children():
            if name == 'model':
                for sub_module in [module.visual.conv1, module.visual.ln_pre, *module.visual.transformer.resblocks]:
                    params_for_layer = {
                        'params': sub_module.parameters(),
                        'lr': hparams[f"{layer_idx}_lr"],
                        'weight_decay': hparams[f"{layer_idx}_wd"]
                    }
                    layerwise_params.append(params_for_layer)
                    layer_idx += 1

        # Classification head of the model
        params_for_layer = {
            'params': model.classification_head.parameters(),
            'lr': hparams[f"{layer_idx}_lr"],
            'weight_decay': hparams[f"{layer_idx}_wd"]
        }
        layerwise_params.append(params_for_layer)
        optimizer = torch.optim.AdamW(layerwise_params)
    else:
        if "lr" not in hparams.keys() and "wd" not in hparams.keys():
            hparams["lr"] = 1e-5
            hparams["wd"] = 0.2
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])
    return optimizer

def _run(index, args):
    start_time = time.time()
    device = xm.xla_device()
    model, preprocess_fn = initialize_model(args)
    model = model.to(device)
    if args.eval_only:
        _mp_evaluate(index, model, args)
        return

    assert args.load_hparams is not None
    with open(args.load_hparams, 'r') as f:
        best_hparams = json.load(f)

    all_datasets = get_datasets(args, model, preprocess_fn)
    train_loader = get_dataloader(all_datasets["id"], is_train=True, args=args, image_encoder=None)
    num_batches = len(train_loader)
    print(f"{num_batches} batches in fine-tuning dataloader")
    total_steps = (args.ft_epochs * num_batches) // xm.xrt_world_size()
    warmup_length = (args.warmup_length * args.accumulation_steps) // xm.xrt_world_size()
    optimizer = create_optimizer(model=model, hparams=best_hparams, layerwise=args.layerwise_opt)
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)

    loss_weights = get_loss_weights(hparams=best_hparams, layerwise=args.layerwise_loss)
    initial_params = [p for p in model.parameters()]
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)
    print_hparams(hparams=best_hparams)
    save_hparams(hparams=best_hparams, args=args)
    set_seed(seed=best_hparams["seed"])

    effective_step = 0
    for epoch in range(0, args.ft_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            step = i + epoch * (num_batches // xm.xrt_world_size())
            scheduler(step)

            batch = maybe_dictionarize(batch)
            print('batch keys', batch.keys())
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            text = batch['metadata'].to(device)
            logits, image_features, text_features, logit_scale = model(images, text)

            try:
                ls = logit_scale[0]
            except Exception:
                ls = logit_scale
            loss = loss_fn(model, logits, labels, image_features, text_features, ls, unlabeled_logits=None)
            loss = loss / args.accumulation_steps
            loss.backward()
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if (i + 1) % args.accumulation_steps == 0:
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
        _mp_evaluate(index, model, args)
        xm.master_print(f"Finished training epoch: {epoch + 1} | Time: {time.time() - start_time} s")


def main(args):
    logger = logging.getLogger('main')
    logger = setup_logging(args, logger)
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