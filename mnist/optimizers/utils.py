import torch
import torch.nn as nn

def get_lopt_net(in_dim, hid_dim=2, out_dim=1):
    return nn.Sequential(
        nn.Linear(in_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim)
    )

def clip_gradient(params, max_norm):
    norm = torch.norm(params.grad.data)
    if norm > max_norm:
        params.grad.data.mul_(max_norm / norm)
    return params


def compute_positional_encoding(tensor_shape):
    if len(tensor_shape) == 1:
        pos_enc = torch.arange(tensor_shape[0]).view(-1, 1)
        # Add a dummy dimension for compatibility with 2D tensors
        pos_enc = torch.cat([pos_enc, torch.zeros_like(pos_enc)], dim=-1)
    else:
        pos_enc = torch.stack(torch.meshgrid([torch.arange(s) for s in tensor_shape]), -1)
        pos_enc = pos_enc.view(-1, pos_enc.shape[-1])
    return pos_enc.float()


def compute_continuous_positional_encoding(tensor_shape, d_model):
    if len(tensor_shape) == 1:
        position = torch.arange(tensor_shape[0]).view(-1, 1)
        # Add a dummy dimension for compatibility with 2D tensors
        position = torch.cat([position, torch.zeros_like(position)], dim=-1)
    else:
        position = torch.stack(torch.meshgrid([torch.arange(s) for s in tensor_shape]), -1)
    # Create a tensor of term denominators for the exponential function. This will generate a geometric progression.
    # This is used to create sinusoids of different wavelengths, allowing the model to learn to attend to both absolute and relative positions.
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

    pos_enc = torch.zeros_like(position)
    pos_enc[:, 0::2] = torch.sin(position[:, 0::2] * div_term)
    pos_enc[:, 1::2] = torch.cos(position[:, 1::2] * div_term)

    return pos_enc.view(-1, pos_enc.shape[-1])