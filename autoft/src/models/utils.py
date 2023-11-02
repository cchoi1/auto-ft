import json
import os
import pickle
import random

import numpy as np
import torch

def get_num_classes(args):
    if args.id in ["sst2Train", "PatchCamelyonTrain"]:
        return 2
    elif args.id in ["ImageNetKShot"]:
        return 1000
    elif args.id == "IWildCamTrain":
        return 294
    elif args.id == "FMOWTrain":
        return 62
    elif args.id == "CIFAR10":
        return 10
    else:
        raise ValueError("Invalid dataset")


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


# code from WiSE-FT utils.py
# import os
#
# import torch
# import pickle
# from tqdm import tqdm
# import math
#
# import numpy as np
#
#
# def assign_learning_rate(param_group, new_lr):
#     param_group["lr"] = new_lr
#
#
# def _warmup_lr(base_lr, warmup_length, step):
#     return base_lr * (step + 1) / warmup_length
#
#
# def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
#     if not isinstance(base_lrs, list):
#         base_lrs = [base_lrs for _ in optimizer.param_groups]
#     assert len(base_lrs) == len(optimizer.param_groups)
#
#     def _lr_adjuster(step):
#         for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
#             if step < warmup_length:
#                 lr = _warmup_lr(base_lr, warmup_length, step)
#             else:
#                 e = step - warmup_length
#                 es = steps - warmup_length
#                 lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
#             assign_learning_rate(param_group, lr)
#
#     return _lr_adjuster
#
#
# def accuracy(output, target, topk=(1, )):
#     pred = output.topk(max(topk), 1, True, True)[1].t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     return [
#         float(correct[:k].reshape(-1).float().sum(0,
#                                                   keepdim=True).cpu().numpy())
#         for k in topk
#     ]
#
#
# def torch_save(classifier, save_path):
#     if os.path.dirname(save_path) != '':
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, 'wb') as f:
#         pickle.dump(classifier.cpu(), f)
#
#
# def torch_load(save_path, device=None):
#     with open(save_path, 'rb') as f:
#         classifier = pickle.load(f)
#     if device is not None:
#         classifier = classifier.to(device)
#     return classifier
#
#
# def fisher_save(fisher, save_path):
#     if os.path.dirname(save_path) != '':
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     fisher = {k: v.cpu() for k, v in fisher.items()}
#     with open(save_path, 'wb') as f:
#         pickle.dump(fisher, f)
#
#
# def fisher_load(save_path, device=None):
#     with open(save_path, 'rb') as f:
#         fisher = pickle.load(f)
#     if device is not None:
#         fisher = {k: v.to(device) for k, v in fisher.items()}
#     return fisher
#
#
# def get_logits(inputs, classifier, classification_head):
#     assert callable(classifier)
#     if hasattr(classifier, 'to'):
#         classifier = classifier.to(inputs.device)
#         classification_head = classification_head.to(inputs.device)
#     feats = classifier(inputs)
#     return classification_head(feats)
#
#
# def get_feats(inputs, classifier):
#     assert callable(classifier)
#     if hasattr(classifier, 'to'):
#         classifier = classifier.to(inputs.device)
#     feats = classifier(inputs)
#     # feats = feats / feats.norm(dim=-1, keepdim=True)
#     return feats
#
#
# def get_probs(inputs, classifier):
#     if hasattr(classifier, 'predict_proba'):
#         probs = classifier.predict_proba(inputs.detach().cpu().numpy())
#         return torch.from_numpy(probs)
#     logits = get_logits(inputs, classifier)
#     return logits.softmax(dim=1)
#
#
# class LabelSmoothing(torch.nn.Module):
#     def __init__(self, smoothing=0.0):
#         super(LabelSmoothing, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#
#     def forward(self, x, target):
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)
#
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()