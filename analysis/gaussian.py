#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal, kl_divergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iris_colors = ["#FF6150", "#134E6F", "#1AC0C6", "#FFA822", "#DEE0E6", "#091A29"]

def set_seed(SEED=1):
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Initialize dimensions and number of data points
N = 10
num_data_points = 100

set_seed()
zeros = torch.zeros(N)
prior_cov = torch.eye(N)
id_vars = torch.ones(N)
id_vars[:4] = 1e-3
id_cov = torch.diag(id_vars)
ood_vars = torch.ones(N)
ood_vars[1:5] = 1e-3
ood_cov = torch.diag(ood_vars)
zeros, prior_cov = zeros.to(device), prior_cov.to(device)
id_vars, id_cov = id_vars.to(device), id_cov.to(device)
ood_vars, ood_cov = ood_vars.to(device), ood_cov.to(device)

# Generate ID data
id_data = MultivariateNormal(zeros, id_cov).sample((num_data_points,))
id_val = MultivariateNormal(zeros, id_cov).sample((num_data_points,))
id_data, id_val = id_data.to(device), id_val.to(device)

# Generate OOD data
ood_data = MultivariateNormal(zeros, ood_cov).sample((num_data_points,))


def dimwise_nll(data, mean, cov):
    D = data.shape[1]
    _nll_vals = [-Normal(mean[d], cov[d].squeeze()).log_prob(data[:, d]) for d in range(D)]
    return torch.stack(_nll_vals, dim=1)

id_vars, ood_vars

#%%
def run_weighted_finetuning(alphas, num_steps=20, prior_type="normal"):
    """ One run of fine-tuning with a dimensionwise weighted NLL. """
    alphas = alphas.to(device).requires_grad_(False)
    cov = torch.ones(N).to(device).requires_grad_(True)
    optimizer = optim.SGD([cov], lr=1.0)

    if prior_type == "normal":
        prior = MultivariateNormal(zeros, prior_cov)
    elif prior_type == "wide":
        wide_cov = torch.diag(torch.ones(N) * 100).to(device)
        prior = MultivariateNormal(zeros, wide_cov)

    metrics = defaultdict(list)
    for step in range(num_steps):
        D = id_data.shape[1]
        cov_exp = cov.exp()
        nll_dimwise = dimwise_nll(id_data, zeros, cov_exp)

        kl_div = kl_divergence(MultivariateNormal(zeros, torch.diag(cov_exp)), prior).mean()

        nll_weighted = (nll_dimwise * alphas).sum()
        loss = nll_weighted + kl_div
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cov, 1.0)
        optimizer.step()

        ood_nll = dimwise_nll(ood_data, zeros, cov_exp)

        metrics["id_nll"].append(nll_dimwise.sum().item())
        metrics["w_id_nll"].append(nll_weighted.item())
        metrics["kl_div"].append(kl_div.item())
        metrics["ood_nll"].append(ood_nll.sum().item())
    metrics["cov"].append(cov.exp().detach().cpu().numpy())
    return metrics

alphas = torch.ones(N) * 0.001
results = run_weighted_finetuning(alphas)
for i in range(0, len(results["id_nll"]), 2):
    print(f"Step {i}, ID NLL: {results['id_nll'][i]:.2f}, weighted ID NLL: {results['w_id_nll'][i]:.2f}, OOD NLL: {results['ood_nll'][i]:.2f}, KL: {results['kl_div'][i]:.2f}")
print(f"Final covariance: {results['cov'][-1]}")

alphas = torch.ones(N) * 0.01
results = run_weighted_finetuning(alphas)
for i in range(0, len(results["id_nll"]), 2):
    print(f"Step {i}, ID NLL: {results['id_nll'][i]:.2f}, weighted ID NLL: {results['w_id_nll'][i]:.2f}, OOD NLL: {results['ood_nll'][i]:.2f}, KL: {results['kl_div'][i]:.2f}")
print(f"Final covariance: {results['cov'][-1]}")

#%%
def objective(trial, dimwise, prior_type):
    set_seed()
    alpha_low, alpha_high = 1e-6, 0.1
    if dimwise:
        alphas = [trial.suggest_float(f'alpha_{i}', alpha_low, alpha_high, log=True) for i in range(N)]
    else:
        alpha = trial.suggest_float('alpha', alpha_low, alpha_high, log=True)
        alphas = [alpha for i in range(N)]
    alphas = torch.Tensor(alphas)
    metrics = run_weighted_finetuning(alphas, prior_type=prior_type)
    return metrics["ood_nll"][-1]

def run_exp(dimwise, prior_type, n_trials=50):
    objective_ = lambda trial: objective(trial, dimwise, prior_type)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_, n_trials=n_trials)
    best_params = study.best_params
    if dimwise:
        best_alpha = [best_params[f'alpha_{i}'] for i in range(N)]
    else:
        best_alpha = [best_params[f'alpha'] for i in range(N)]
    values = [t.values[0] for t in study.get_trials()]
    values_min = np.minimum.accumulate(values)
    return {"best_alpha": best_alpha, "values": values, "values_min": values_min}


all_results = {}
all_results["dim_normal"] = run_exp(True, "normal")
all_results["all_normal"] = run_exp(False, "normal")
all_results["dim_wide"] = run_exp(True, "wide")
all_results["all_wide"] = run_exp(False, "wide")

#%%
# outer-loop learning curve plot
dim_values1 = all_results["dim_normal"]["values"]
all_values1 = all_results["all_normal"]["values"]
dim_values_min1 = np.minimum.accumulate(dim_values1)
all_values_min1 = np.minimum.accumulate(all_values1)

dim_values2 = all_results["dim_wide"]["values"]
all_values2 = all_results["all_wide"]["values"]
dim_values_min2 = np.minimum.accumulate(dim_values2)
all_values_min2 = np.minimum.accumulate(all_values2)

plt.figure(figsize=(7, 3))

# First subplot
plt.subplot(1, 2, 1)
plt.scatter(range(len(dim_values1)), dim_values1, color=iris_colors[0], alpha=0.1)
plt.scatter(range(len(all_values1)), all_values1, color=iris_colors[1], alpha=0.1)
plt.plot(dim_values_min1, "-", color=iris_colors[0], label="Dim-wise Weight")
plt.plot(all_values_min1, "-", color=iris_colors[1], label="Global Weight")
plt.xlabel("Optuna Trials")
plt.ylabel("OOD Data NLL")
plt.yscale("log")
plt.title("Good Prior")

# Second subplot
plt.subplot(1, 2, 2, sharey=plt.gca())
plt.scatter(range(len(dim_values2)), dim_values2, color=iris_colors[0], alpha=0.1)
plt.scatter(range(len(all_values2)), all_values2, color=iris_colors[1], alpha=0.1)
plt.plot(dim_values_min2, "-", color=iris_colors[0], label="Dim-wise Weight")
plt.plot(all_values_min2, "-", color=iris_colors[1], label="Global Weight")
plt.legend(loc="upper right")
plt.xlabel("Optuna Trials")
plt.yticks([])
plt.yscale("log")
plt.title("Misspecified Prior")

plt.tight_layout()
plt.show()

#%%
dim_best_alpha = all_results["dim_normal"]["best_alpha"]
all_best_alpha = all_results["all_normal"]["best_alpha"]
dim_results = run_weighted_finetuning(torch.Tensor(dim_best_alpha))
dim_average_alpha = torch.ones(N) * np.mean(dim_best_alpha)
dim_averaged_results = run_weighted_finetuning(dim_average_alpha)
all_results = run_weighted_finetuning(torch.Tensor(all_best_alpha))
dim_id_nll = dim_results["id_nll"]
dim_ood_nll = dim_results["ood_nll"]
dim_avg_id_nll = dim_averaged_results["id_nll"]
dim_avg_ood_nll = dim_averaged_results["ood_nll"]
all_id_nll = all_results["id_nll"]
all_ood_nll = all_results["ood_nll"]

plt.figure(figsize=(5, 3))
plt.plot(dim_avg_id_nll, "-", color=iris_colors[3], label="Dim-wise Avg")
plt.plot(dim_avg_ood_nll, "--", color=iris_colors[3])
plt.plot(all_id_nll, "-", color=iris_colors[1], label="Global")
plt.plot(all_ood_nll, "--", color=iris_colors[1])
plt.plot(dim_id_nll, "-", color=iris_colors[0], label="Dim-wise")
plt.plot(dim_ood_nll, "--", color=iris_colors[0])
plt.xlabel("Inner Loop Steps")
plt.ylabel("NLL")
plt.yscale("log")
plt.title(f"Evaluation of Learned Weights")
plt.legend()

# %%
dim_best_alpha = np.array(dim_best_alpha)
alpha_splits = {
    "ID": dim_best_alpha[0],
    "Both": dim_best_alpha[1:4].mean(),
    "OOD": dim_best_alpha[4],
    "None": dim_best_alpha[5:].mean()
}
alpha_stds = {
    "ID": 0.0,
    "Both": dim_best_alpha[1:4].std(),
    "OOD": 0.0,
    "None": dim_best_alpha[5:].std()
}

# Bar plot
plt.figure(figsize=(4, 3))
plt.bar(alpha_splits.keys(), alpha_splits.values(), color=iris_colors, yerr=alpha_stds.values(), capsize=5, alpha=0.8, edgecolor="black")
plt.title("Learned Weights")
plt.yscale("log")
# %%
