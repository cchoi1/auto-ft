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

# Initialize dimensions and number of data points
N = 10
num_data_points = 100

# Generate Prior
prior_mean = torch.zeros(N)
prior_cov = torch.eye(N)
prior_mean, prior_cov = prior_mean.to(device), prior_cov.to(device)
prior = MultivariateNormal(prior_mean, prior_cov)

# Generate ID data
id_vars = torch.ones(N)
id_vars[:4] = 1e-3
id_cov = torch.diag(id_vars)
id_vars, id_cov = id_vars.to(device), id_cov.to(device)
id_data = MultivariateNormal(prior_mean, id_cov).sample((num_data_points,))
id_val = MultivariateNormal(prior_mean, id_cov).sample((num_data_points,))
id_data, id_val = id_data.to(device), id_val.to(device)

# Generate OOD data
ood_vars = torch.ones(N)
ood_vars[1:5] = 1e-3
ood_cov = torch.diag(ood_vars)
ood_vars, ood_cov = ood_vars.to(device), ood_cov.to(device)
ood_data = MultivariateNormal(prior_mean, ood_cov).sample((num_data_points,))


def dimwise_nll(data, mean, cov):
    D = data.shape[1]
    _nll_vals = [-Normal(mean[d], cov[d].squeeze()).log_prob(data[:, d]) for d in range(D)]
    return torch.stack(_nll_vals, dim=1)

id_vars, ood_vars

#%%
def run_weighted_finetuning(alphas, num_steps=20):
    """ One run of fine-tuning with a dimensionwise weighted NLL. """
    alphas = alphas.to(device).requires_grad_(False)
    cov = torch.ones(N).to(device).requires_grad_(True)
    optimizer = optim.SGD([cov], lr=1.0)

    metrics = defaultdict(list)
    for step in range(num_steps):
        D = id_data.shape[1]
        cov_exp = cov.exp()
        nll_dimwise = dimwise_nll(id_data, prior_mean, cov_exp)
        kl_div = kl_divergence(MultivariateNormal(prior_mean, torch.diag(cov_exp)), prior).mean()

        nll_weighted = (nll_dimwise * alphas).sum()
        loss = nll_weighted + kl_div
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cov, 1.0)
        optimizer.step()

        ood_nll = dimwise_nll(ood_data, prior_mean, cov_exp)

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
def objective(trial, dimwise):
    alpha_low, alpha_high = 1e-6, 0.1
    if dimwise:
        alphas = [trial.suggest_float(f'alpha_{i}', alpha_low, alpha_high, log=True) for i in range(N)]
    else:
        alpha = trial.suggest_float('alpha', alpha_low, alpha_high, log=True)
        alphas = [alpha for i in range(N)]
    alphas = torch.Tensor(alphas)
    metrics = run_weighted_finetuning(alphas)
    return metrics["ood_nll"][-1]

objective_dim = lambda trial: objective(trial, True)
objective_all = lambda trial: objective(trial, False)

study_dim = optuna.create_study(direction="minimize")
study_dim.optimize(objective_dim, n_trials=50)
dim_best_params = study_dim.best_params
dim_best_alpha = [dim_best_params[f'alpha_{i}'] for i in range(N)]
print(f"Best value: {study_dim.best_value}")
print(f"Optimized alpha values: {dim_best_alpha}")

study_all = optuna.create_study(direction="minimize")
study_all.optimize(objective_all, n_trials=50)
all_best_params = study_all.best_params
all_best_alpha = [all_best_params[f'alpha'] for i in range(N)]
print(f"Best value: {study_all.best_value}")
print(f"Optimized alpha values: {all_best_alpha}")

#%%
# outer-loop learning curve plot
dim_values = [t.values[0] for t in study_dim.get_trials()]
all_values = [t.values[0] for t in study_all.get_trials()]
plt.figure(figsize=(5, 3))
plt.plot(all_values, "o-", color=iris_colors[1], label="Global Weight")
plt.plot(dim_values, "o-", color=iris_colors[0], label="Dim-wise Weight")
plt.legend()
plt.xlabel("Optuna Trials")
plt.ylabel("OOD Data NLL")
plt.yscale("log")
plt.title("Hyperopt Learning Progress")

#%%
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
