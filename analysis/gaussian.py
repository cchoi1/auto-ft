#%%
import optuna
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence, Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dimensions and number of data points
N = 10
num_data_points = 100

# Generate Prior
prior_mean = torch.zeros(N)
prior_cov = torch.eye(N)
prior_mean, prior_cov = prior_mean.to(device), prior_cov.to(device)
prior = MultivariateNormal(prior_mean, prior_cov)

# Generate ID data
rand_idxs = torch.randperm(N)
id_vars = torch.Tensor([0.1 if rand_idxs[i] < N//2 else 1.0 for i in range(N)])
id_cov = torch.diag(id_vars)
id_vars, id_cov = id_vars.to(device), id_cov.to(device)
id_data = MultivariateNormal(prior_mean, id_cov).sample((num_data_points,))
id_val = MultivariateNormal(prior_mean, id_cov).sample((num_data_points,))
id_data, id_val = id_data.to(device), id_val.to(device)

# Generate OOD data
rand_idxs = torch.randperm(N)
ood_vars = torch.Tensor([0.1 if rand_idxs[i] < N//2 else 1.0 for i in range(N)])
ood_cov = torch.diag(ood_vars)
ood_vars, ood_cov = ood_vars.to(device), ood_cov.to(device)
ood_data = MultivariateNormal(prior_mean, ood_cov).sample((num_data_points,))


def dimwise_nll(data, mean, cov):
    D = data.shape[1]
    _nll_vals = [-Normal(mean[d], cov[d].squeeze()).log_prob(data[:, d]) for d in range(D)]
    return torch.stack(_nll_vals, dim=1)


#%%
alphas = torch.ones(N).to(device)
alphas = id_vars.to(device)
cov = torch.ones(N).to(device)
cov.requires_grad = True
optimizer = optim.SGD([cov], lr=5e-3)

# Training loop for Maximum Likelihood Estimation with hyperparameters
for epoch in range(100):
    optimizer.zero_grad()

    # Negative log likelihood for ID data
    D = id_data.shape[1]
    nll_dimwise = []
    cov_exp = torch.exp(cov)
    for d in range(D):
        cov_d = cov_exp[d].squeeze()
        nll_d = -Normal(prior_mean[d], cov_d).log_prob(id_data[:, d])
        nll_dimwise.append(nll_d)
    nll_dimwise = torch.stack(nll_dimwise, dim=1)
    kl_div = kl_divergence(MultivariateNormal(prior_mean, torch.diag(cov_exp)), prior).mean()

    nll_weighted = (nll_dimwise * alphas).sum()

    loss = nll_weighted + kl_div
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, NLL_w: {nll_weighted.item()}, KL: {kl_div.item()}")

# Print the optimized hyperparameters and covariance matrix
print("Optimized Covariance:", cov_exp.detach() ** 2)
nll_dimwise_val = dimwise_nll(id_val, prior_mean, cov_exp)
print("NLL on validation data:", nll_dimwise_val.sum().item())


#%%
# Objective function to optimize
def objective(trial):
    # Suggest values for alpha
    alpha_suggested = [trial.suggest_float(f'alpha_{i}', 0.0, 1.0) for i in range(N)]
    # lr_suggested = trial.suggest_float('lr', 1e-4, 1e-2)

    alphas = torch.Tensor(alpha_suggested).to(device).requires_grad_(False)
    cov = torch.ones(N).to(device).requires_grad_(True)

    # Initialize optimizer
    optimizer = optim.SGD([cov], lr=5e-3)

    # Training loop for Maximum Likelihood Estimation with hyperparameters
    for epoch in range(10):
        cov_exp = torch.exp(cov)
        nll_dimwise_id = dimwise_nll(id_data, prior_mean, cov_exp)
        nll_weighted = (nll_dimwise_id * alphas).sum()
        prior_kl = kl_divergence(MultivariateNormal(prior_mean, torch.diag(cov_exp)), prior).mean()

        inner_objective = nll_weighted + prior_kl
        inner_objective.backward()
        optimizer.step()
        optimizer.zero_grad()

    nll_dimwise_ood = dimwise_nll(id_val, prior_mean, cov_exp)
    meta_objective = (nll_dimwise_ood).sum()
    return meta_objective.item()

# Initialize dimensions and number of data points
N = 10
D = 10

# Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

# Extract the optimized alpha
best_params = study.best_params
opt_alpha = [best_params[f'alpha_{i}'] for i in range(N)]
print(f"Optimized alpha values: {opt_alpha}")

# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming ood_vars and opt_alpha are lists of length N
variances_np = id_vars.detach().cpu().numpy()
opt_alpha_np = np.array(opt_alpha)

# Create an index for each group
barWidth = 0.3
r1 = np.arange(len(variances_np))
r2 = [x + barWidth for x in r1]

# Create the bar plot
plt.bar(r1, variances_np, color='b', width=barWidth, edgecolor='grey', label='ood_vars')
plt.bar(r2, opt_alpha_np, color='r', width=barWidth, edgecolor='grey', label='opt_alpha')

# Add labels and title
plt.xlabel('Group', fontweight='bold')
plt.ylabel('Value', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(variances_np))], ['Group'+str(i) for i in range(1, 11)])
plt.legend()

# Show the plot
plt.show()
# %%
