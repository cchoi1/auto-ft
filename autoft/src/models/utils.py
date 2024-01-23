import json
import os
import pickle
import random

import numpy as np
import torch

NUM_DATASET_CLASSES = {
    "sst2Train": 2,
    "PatchCamelyonTrain": 2,
    "Caltech101Train": 101,
    "Flowers102Train": 102,
    "IWildCamTrain": 294,
    "FMOWTrain": 62,
    "ImageNet": 1000,
    "CIFAR10": 10
}

TEST_METRICS = {
    "sst2": "sst2Test:top1",
    "PatchCamelyon": "PatchCamelyonTest:top1",
    "IWildCam": "IWildCamOODTest:F1-macro_all",
    "FMOW": "FMOWOODTest:acc_worst_region",
    "ImageNet": "ImageNet:top1",
    "ImageNet4": "ImageNet4:top1",
    "ImageNet16": "ImageNet16:top1",
    "ImageNet32": "ImageNet32:top1",
    "Caltech101": "Caltech101Test:top1",
    "Flowers102": "Flowers102Test:top1",
    "StanfordCars": "StanfordCarsTest:top1"
}

VAL_METRICS = {
    "IWildCam": "IWildCamIDVal:F1-macro_all",
    "FMOW": "FMOWIDVal:acc_worst_region",
    "ImageNet": "ImageNet:top1",
    "ImageNet4": "ImageNet4:top1",
    "ImageNet16": "ImageNet16:top1",
    "ImageNet32": "ImageNet32:top1",
    "sst2": "sst2ValEarlyStopping:top1",
    "PatchCamelyon": "PatchCamelyonValEarlyStopping:top1",
    "Caltech101": "Caltech101ValEarlyStopping:top1",
    "Flowers102": "Flowers102ValEarlyStopping:top1",
    "StanfordCars": "StanfordCarsValEarlyStopping:top1",
    "CIFAR10": "CIFAR10:top1"
}


def get_num_classes(args):
    try:
        return NUM_DATASET_CLASSES[args.id]
    except KeyError:
        raise ValueError("Invalid dataset")


def test_metric_str(args):
    for key in TEST_METRICS:
        if key in args.id:
            return TEST_METRICS[key]
    raise ValueError("Invalid dataset for test metric")


def val_metric_str(args):
    for key in VAL_METRICS:
        if key in args.id:
            return VAL_METRICS[key]
    raise ValueError("Invalid dataset for validation metric")


def print_hparams(hparams):
    print("\nHyperparameters:")
    for key, value in hparams.items():
        if not "dataw" in key:
            print(f"{key}: {value}")


def save_hparams(hparams, args):
    save_file = os.path.join(args.save, 'hparams.json')
    os.makedirs(args.save, exist_ok=True)
    print(f"\nSaving hyperparameters to {save_file}.")
    hparams["seed"] = int(hparams["seed"])
    if "ce" in args.losses and "lossw_ce" not in hparams.keys(): # Save cross-entropy loss weight
        hparams["lossw_ce"] = 1.0
    with open(save_file, 'w') as f:
        json.dump(hparams, f)


def print_train_update(logger, print_every, total_steps, step, loss, batch_time):
    should_print = (print_every is not None and step % print_every == 0)
    if should_print:
        percent_complete = 100 * step / total_steps
        print(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
            f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}")
        logger.info(f"Train Iter: {step}/{total_steps} [{percent_complete:.0f}% ]\t"
                    f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}")


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


def get_subset(dataset, num_datapoints):
    rand_idxs = torch.randperm(len(dataset))[:num_datapoints]
    return torch.utils.data.Subset(dataset, rand_idxs)


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
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
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