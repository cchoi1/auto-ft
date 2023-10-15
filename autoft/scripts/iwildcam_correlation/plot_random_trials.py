# %%
import collections
import json
import os
from glob import glob

import pandas as pd
import seaborn as sns

def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def plot_random_trials(runs_dir):
    trial_files = glob(os.path.join(runs_dir, "trial_*.json"))
    trials = []
    for f in trial_files:
        with open(f, "r") as f:
            trial = json.load(f)
        trial = flatten(trial)
        trials.append(trial)

    aggregated_metrics = {}
    for trial in trials:
        for metric, value in trial.items():
            if metric not in aggregated_metrics:
                aggregated_metrics[metric] = [value]
            else:
                aggregated_metrics[metric].append(value)

    aggregated_metrics
    print(aggregated_metrics.keys())
    pd.DataFrame(aggregated_metrics)

    Ns = [3, 10, 30, 100, 300]
    metrics = ["F1-macro_all"]
    selected_metrics = [f"{n}_{metric}" for n in Ns for metric in metrics]
    df_selected_metrics = pd.DataFrame(aggregated_metrics)[selected_metrics]
    df_selected_metrics
    sns.pairplot(df_selected_metrics)
    print(runs_dir)

runs_dirs = glob("../../logs/saved/IWildCamTrain/autoft/*/*_is10_*")
for runs_dir in runs_dirs:
    try:
        plot_random_trials(runs_dir)
    except Exception as e:
        print(e)

# %%
