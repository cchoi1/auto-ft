import json
import logging
import os
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data.distributed import DistributedSampler

import src.datasets as datasets
from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize
from src.logger import setup_logging
from src.losses.learnedloss import LearnedLoss
from src.models.eval import _mp_evaluate
from src.models.modeling import ImageClassifier
from src.models.utils import cosine_lr, set_seed, print_hparams, save_hparams, val_metric_str, test_metric_str
from src.datasets.laion import get_data
from src.models.zeroshot import get_zeroshot_classifier


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
        xm.master_print(f"Using distributed sampler with {xm.xrt_world_size()} replicas")
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
    torch.distributed.init_process_group(
        backend='xla',
        world_size=xm.xrt_world_size(),
        rank=xm.get_ordinal()
    )

    device = xm.xla_device()
    model, preprocess_fn = initialize_model(args)
    model = model.to(device)
    xm.master_print(f"Got model in {time.time() - start_time} s")
    if args.eval_only:
        if not args.no_regenerate_head:
            with torch.no_grad():
                classification_head = get_zeroshot_classifier(args, model.image_encoder.model)
                classification_head = classification_head.cuda()
        else:
            classification_head = model.classification_head
        _mp_evaluate(index, model, classification_head, args)
        return

    assert args.load_hparams is not None
    with open(args.load_hparams, 'r') as f:
        best_hparams = json.load(f)

    dataset_start_time = time.time()
    all_datasets = get_datasets(args, model, preprocess_fn)
    xm.master_print(f"Finished fetching fine-tuning dataset in {time.time() - dataset_start_time:.2f} s")
    dataloader_start_time = time.time()
    train_loader = all_datasets["id"]["train_ft"].dataloader
    xm.master_print(f"Finished creating fine-tuning dataloader in {time.time() - dataloader_start_time:.2f} s")
    xm.master_print(f"Size of fine-tuning dataset: {all_datasets['id']['train_ft'].dataloader.num_samples} samples")
    num_batches_per_epoch = all_datasets['id']["train_ft"].dataloader.num_samples // (args.batch_size * xm.xrt_world_size())
    xm.master_print(f"{num_batches_per_epoch} batches in fine-tuning dataset...")
    total_steps = args.ft_epochs * num_batches_per_epoch
    warmup_length = (args.warmup_length * args.accumulation_steps) // xm.xrt_world_size()
    optimizer = create_optimizer(model=model, hparams=best_hparams, layerwise=args.layerwise_opt)
    scheduler = cosine_lr(optimizer, args.lr, warmup_length, total_steps)

    loss_weights = get_loss_weights(hparams=best_hparams, layerwise=args.layerwise_loss)
    initial_params = [p for p in model.parameters()]
    loss_fn = LearnedLoss(losses=args.losses, loss_weights=loss_weights, initial_params=initial_params)
    print_hparams(hparams=best_hparams)
    save_hparams(hparams=best_hparams, args=args)
    set_seed(seed=best_hparams["seed"])

    val_metrics = {}
    best_val_metric = -float('inf')
    model.train()
    for epoch in range(0, args.ft_epochs):
        epoch_start_time = time.time()
        for i, batch in enumerate(train_loader):
            step_start_time = time.time()
            step = i + epoch * num_batches_per_epoch
            scheduler(step)

            batch = maybe_dictionarize(batch)
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
            if (i + 1) % args.accumulation_steps == 0:
                if args.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                xm.optimizer_step(optimizer)

            xm.master_print(f"Epoch [{epoch + 1}/{args.ft_epochs}], Step [{step}/{total_steps}], Loss: {loss.item():.2f}, Step Time: {time.time() - step_start_time:.2f} s")
        # Save the current checkpoint along with optimizer and scheduler
        if epoch == args.ft_epochs - 1:
            if xm.is_master_ordinal():
                model_save_path = os.path.join(args.save, f"model_{epoch}.pt")
                model = model.to('cpu')
                model.save(model_save_path)
                model = model.to(device)
                xm.master_print("Saved model to", model_save_path)
            ckpt_save_path = os.path.join(args.save, f"opt_sched_{epoch}.pt")
            xm.save({
                'optimizer_state': optimizer.state_dict(),
                'scheduler_params': {
                    'base_lrs': args.lr,
                    'warmup_length': args.warmup_length,
                    'steps': total_steps,
                    'current_step': step
                },
                'epoch': epoch
            }, ckpt_save_path)
            xm.master_print(f"Saved optimizer and scheduler to {ckpt_save_path}")
        # Evaluate
        # _mp_evaluate(index, model, args)
        if not args.no_regenerate_head:
            with torch.no_grad():
                classification_head = get_zeroshot_classifier(args, model.module.image_encoder.model)
                classification_head = classification_head.cuda()
        else:
            classification_head = model.module.classification_head
        eval_results = _mp_evaluate(model, classification_head, args)
        val_metrics[epoch] = eval_results
        curr_val_metric = eval_results[val_metric_str(args)]
        if curr_val_metric > best_val_metric:
            print(f"Best val metric of {curr_val_metric} at epoch {epoch}.")
            best_val_metric = curr_val_metric
        print(f"Epoch {epoch}, Time: {time.time() - epoch_start_time:.3f}")
        xm.master_print(f"Finished training epoch: {epoch + 1} | Time: {time.time() - start_time:.2f} s")


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