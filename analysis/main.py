#%%
from collections import defaultdict
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def rbf_kernel(x1, x2, variance=0.005):
    return exp(-1 * ((x1-x2) ** 2) / (2*variance))

def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

def set_random(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_random()
xs = np.arange(-1.0, 1.0, 0.001)
true_y = xs * 2 - 0.3

mean = np.zeros(len(xs))
gram = gram_matrix(xs)
get_gp_sample = lambda: np.random.multivariate_normal(mean, gram) * 0.3

id_delta = get_gp_sample()
id_y = true_y + id_delta

gp_samples = [get_gp_sample() for _ in range(5)]
ood_ys = [true_y + gp_sample for gp_sample in gp_samples]

for y in ood_ys:
    plt.plot(xs, y, "b", alpha=0.2)
plt.plot(xs, true_y, "--", c="k", linewidth=2)
plt.plot(xs, id_y, "r", linewidth=1)

#%%
N = len(xs)
all_idxs = np.arange(N)
np.random.shuffle(all_idxs)
num_train = int(N*0.8)
train_idxs, test_idxs = all_idxs[:num_train], all_idxs[num_train:]
print(len(train_idxs), len(test_idxs))
tr_x, tr_y = xs[train_idxs].reshape(-1, 1), id_y[train_idxs]
test_x, test_y = xs[test_idxs].reshape(-1, 1), id_y[test_idxs]
tr_x, tr_y = torch.tensor(tr_x).cuda().float(), torch.tensor(tr_y).cuda().float()
test_x, test_y = torch.tensor(test_x).cuda().float(), torch.tensor(test_y).cuda().float()
ood_ys = [torch.tensor(ood_y).cuda() for ood_y in ood_ys]

all_results = {}
for wd in [0.0, 1e-2, 1e-1, 1.0]:
    set_random()
    D = 1024
    net = torch.nn.Sequential(nn.Linear(1, D), nn.ReLU(), nn.Linear(D, 1))
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=wd)
    loss_fn = nn.MSELoss()
    metrics = defaultdict(list)
    for i in range(1, 2001):
        opt.zero_grad()
        out = net(tr_x).squeeze()
        loss = loss_fn(out, tr_y)
        loss.backward()
        opt.step()

        test_preds = net(test_x).squeeze()
        id_loss = loss_fn(test_preds, test_y)
        ood_losses = [loss_fn(test_preds, ood_y[test_idxs]) for ood_y in ood_ys]
        ood_loss = torch.stack(ood_losses).mean()

        metrics["train_loss"].append(loss.item())
        metrics["id_loss"].append(id_loss.item())
        metrics["ood_loss"].append(ood_loss.item())
    all_results[wd] = metrics

    # plot predictions
    preds = net(torch.tensor(xs).cuda().float().reshape(-1, 1)).squeeze().detach().cpu().numpy()
    plt.plot(xs, preds, lw=2, label=f"wd={wd}", alpha=0.7)
    print(loss.item(), id_loss.item(), ood_loss.item())
plt.plot(xs, true_y, "--", c="k", linewidth=2)
plt.plot(xs, id_y, "gray", linewidth=1)
plt.legend()
plt.title("Final Predictions")
plt.show()


wd_vals = list(all_results.keys())
best_id_losses = [min(all_results[wd]['id_loss']) for wd in wd_vals]
last_id_losses = [all_results[wd]['id_loss'][-1] for wd in wd_vals]
best_ood_losses = [min(all_results[wd]['ood_loss']) for wd in wd_vals]
last_ood_losses = [all_results[wd]['ood_loss'][-1] for wd in wd_vals]
earlystop_ood_losses = [all_results[wd]['ood_loss'][np.argmin(all_results[wd]['id_loss'])] for wd in wd_vals]

# Bar Plot
labels = wd_vals
width = 0.2  # width of the bars
x = np.arange(len(labels))

fig, ax = plt.subplots()

# rects1 = ax.bar(x - width, best_id_losses, width, label='Best ID Loss', alpha=0.6, edgecolor='black')
rects1 = ax.bar(x - width, last_id_losses, width, label='Last ID Loss', alpha=0.6, edgecolor='black')
rects2 = ax.bar(x, last_ood_losses, width, label='Last OOD Loss', alpha=0.6, edgecolor='black')
rects3 = ax.bar(x + width, earlystop_ood_losses, width, label='ID Early stop OOD Loss', alpha=0.6, edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Losses')
ax.set_title('Effect of Weight Decay on Losses')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0.0, 0.4)
plt.xlabel("Weight Decay")

plt.show()
# %%
