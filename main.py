#%%
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from sine import generate_sine_data


def get_network(hidden=50):
    net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1))
    return net


def pretrain_net(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = get_network()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    data = generate_sine_data(N=1000)
    train_x, train_y = data["train"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)

    loss_fn = nn.MSELoss()
    for i in range(3000):
        preds = net(train_x)
        loss = loss_fn(preds, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Final loss: {loss.item():.4f}")
    return net


ckpt_path = Path("/iris/u/yoonho/robust-optimizer/ckpts/sine")
ckpt_path.mkdir(exist_ok=True)

for seed in range(100):
    filename = ckpt_path / f"pretrain_{seed}.pt"
    if not filename.exists():
        net = pretrain_net(seed=seed)
        torch.save(net.state_dict(), filename)
        print(f"Saved pretrained net to {filename}!")

# %% Start meta-learning
pretrained_net = get_network()
pretrained_net.load_state_dict(torch.load(ckpt_path / "pretrain_0.pt"))
# TODO
