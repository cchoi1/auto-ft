import importlib
import os
import random
from collections import OrderedDict

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lopt_info(net, args):
    if args.features is not None:
        input_dim = len(args.features)
        if "pos_enc_cont" in args.features:
            input_dim += 1
        if "pos_enc" in args.features:
            input_dim += 1
        if "iter" in args.features:
            input_dim += 8
        if "momentum" in args.features:
            input_dim += 4
    else:
        input_dim = len([p for p in net.parameters()])

    if args.output == "update":
        output_dim = 2
    else:
        output_dim = 1
        if args.wnb:
            output_dim += 1
        if args.momentum:
            output_dim += 1

    meta_params_start_idx = 0
    meta_params_end_idx = input_dim * output_dim - 1
    if args.loss_name is not None:
        lloss_info = get_lloss_info(net, args)
        meta_params_start_idx = lloss_info["meta_params"]["end"]
        meta_params_end_idx = meta_params_start_idx + input_dim * output_dim - 1

    lopt_info = {
        "features": args.features,
        "input_dim": input_dim,
        "hidden_dim": args.lopt_net_dim,
        "output_dim": output_dim,
        "tensor_shapes": [p.data.shape for p in net.parameters()],
        "wnb": args.wnb,
        "momentum": args.momentum,
        "output": args.output,
        "meta_params": {
            "start": meta_params_start_idx,
            "end": meta_params_end_idx,
        }
    }
    return lopt_info

def get_lloss_info(net, args):
    if args.loss_name is None:
        return {}
    loss_module = importlib.import_module(f"losses.{args.loss_name.lower()}")
    loss_fn = getattr(loss_module, args.loss_name)
    lloss_info = {
        "meta_params": {
            "start": 0,
            "end": loss_fn.get_num_meta_params(net) - 1,
        }
    }
    return lloss_info

def get_per_layer_parameters(model):
    grouped_parameters = OrderedDict()
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]  # Get the layer name from the full parameter name
        if layer_name not in grouped_parameters:
            grouped_parameters[layer_name] = []
        grouped_parameters[layer_name].append(param)
    return grouped_parameters


def get_lr(step, warmup_steps, base_lr):
    """Compute learning rate with warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_meta_params(opt_trainer, exp_name: str, meta_step: int):
    fn = f"results/{exp_name}/{meta_step}.npy"
    if type(opt_trainer.meta_params) == torch.Tensor:
        meta_params = opt_trainer.meta_params.cpu().detach().numpy()
        np.save(fn, np.array(meta_params))
    else:
        meta_params = [per_param_mp.cpu().detach().numpy() for per_param_mp in opt_trainer.meta_params]
        np.savez(fn, *meta_params)