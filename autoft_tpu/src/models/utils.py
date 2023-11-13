import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import json

def test_metric_str(args):
    if "sst2" in args.id:
        metric = "sst2Test:top1"
    elif "PatchCamelyon" in args.id:
        metric = "PatchCamelyonTest:top1"
    elif "ImageNet" in args.id:
        metric = f"ImageNet{args.k}Shot:top1"
    elif "IWildCam" in args.id:
        metric = "IWildCamOODTest:F1-macro_all"
    elif "FMOW" in args.id:
        metric = "FMOWOODTest:acc_worst_region"
    elif "Caltech101" in args.id:
        metric = "Caltech101Test:top1"
    elif "Flowers102" in args.id:
        metric = "Flowers102Test:top1"
    elif "StanfordCars" in args.id:
        metric = "StanfordCarsTest:top1"
    return metric


def val_metric_str(args):
    if "IWildCam" in args.id:
        metric = "IWildCamIDVal:F1-macro_all"
    elif "FMOW" in args.id:
        metric = "FMOWIDVal:acc_worst_region"
    elif "ImageNet" in args.id:
        metric = "ImageNet:top1"
    elif "sst2" in args.id:
        metric = "sst2ValEarlyStopping:top1"
    elif "PatchCamelyon" in args.id:
        metric = "PatchCamelyonValEarlyStopping:top1"
    elif "Caltech101" in args.id:
        metric = "Caltech101ValEarlyStopping:top1"
    elif "Flowers102" in args.id:
        metric = "Flowers102ValEarlyStopping:top1"
    elif "StanfordCars" in args.id:
        metric = "StanfordCarsValEarlyStopping:top1"
    return metric

def _get_device_spec(device):
  ordinal = xm.get_ordinal(defval=-1)
  return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


def now(format='%H:%M:%S'):
  return datetime.now().strftime(format)


def print_train_update(device, tracker, loss, step, total_steps, epoch=None):
  """Prints the training metrics at a given step.

  Args:
    device (torch.device): The device where these statistics came from.
    step_num (int): Current step number.
    loss (float): Current loss.
    rate (float): The examples/sec rate for the current batch.
    global_rate (float): The average examples/sec rate since training began.
    epoch (int, optional): The epoch number.
  """
  update_data = [
      'Training', 'Device={}'.format(_get_device_spec(device)),
      'Epoch={}'.format(epoch) if epoch is not None else None,
      'Step={} / {}'.format(step, total_steps), 'Loss={:.5f}'.format(loss),
      'Rate={:.2f}'.format(tracker.rate()), 'GlobalRate={:.2f}'.format(tracker.global_rate()),
      'Percent Completed={:.2f}'.format(step / total_steps),
      'Time={}'.format(now())
  ]
  xm.master_print('|', ' '.join(item for item in update_data if item), flush=True)


def setup_dataloader(dataloader, device):
    """Configure the dataloader for multi-core TPU or multi-GPU if available."""
    if is_tpu_available():
        dataloader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
    return dataloader


def is_tpu_available():
    return len(xm.get_xla_supported_devices()) >= 8


def extract_from_data_parallel(model):
    if isinstance(model, torch.nn.DataParallel):
        return next(model.children())
    return model


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_hparams(hparams):
    xm.master_print("\nHyperparameters:")
    for key, value in hparams.items():
        if not "dataw" in key:
            xm.master_print(f"{key}: {value}")


def save_hparams(hparams, args):
    save_file = os.path.join(args.save, 'hparams.json')
    os.makedirs(args.save, exist_ok=True)
    xm.master_print(f"\nSaving hyperparameters to {save_file}.")
    hparams["seed"] = int(hparams["seed"])
    if "ce" in args.losses and "lossw_ce" not in hparams.keys(): # Save cross-entropy loss weight
        hparams["lossw_ce"] = 1.0
    with open(save_file, 'w') as f:
        json.dump(hparams, f)


def get_subset(dataset, num_datapoints):
    rand_idxs = torch.randperm(len(dataset))[:num_datapoints]
    return torch.utils.data.Subset(dataset, rand_idxs)


def save_model(args, model, logger):
    os.makedirs(args.save, exist_ok=True)
    model_path = os.path.join(args.save, f'checkpoint_{args.ft_epochs}.pt')
    xm.master_print('Saving model to', model_path)
    logger.info(f"Saving model to {model_path}")
    model.module.save(model_path)
    return model_path


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)
    with open(save_path, 'wb') as f:
        torch.save(classifier.cpu(), f)


# def torch_load(save_path, device=None):
#     with open(save_path, 'rb') as f:
#         classifier = pickle.load(f)
#     if device is not None:
#         classifier = classifier.to(device)
#     return classifier

def torch_load(save_path, device=None):
    cpu_device = torch.device('cpu')
    tpu_device = xm.xla_device()
    xm.master_print(f"Loading from device {cpu_device} and putting on device {tpu_device}")
    classifier = torch.load(save_path, map_location=cpu_device)
    classifier = classifier.to(tpu_device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_logits_encoder(inputs, encoder, classification_head):
    assert callable(encoder)
    if hasattr(encoder, 'to'):
        encoder = encoder.to(inputs.device)
        classification_head = classification_head.to(inputs.device)
    feats = encoder(inputs)
    return classification_head(feats)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()