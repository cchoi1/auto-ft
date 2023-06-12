from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_sine_data(
    train_range=(0, 2 * np.pi),
    test_range=(0, 2 * np.pi),
    N=100,
    test_N=100,
    amplitude=1,
    phase=0,
    bias=0,
    noise_std=0.05,
):
    """Generate one train-test split of sine task."""
    train_x = np.random.uniform(low=train_range[0], high=train_range[1], size=N)
    test_x = np.random.uniform(low=test_range[0], high=test_range[1], size=test_N)
    f = lambda x: amplitude * np.sin(x + phase) + bias
    train_y = f(train_x) + np.random.normal(loc=0, scale=noise_std, size=train_x.shape)
    test_y = f(test_x)
    return {"train": (train_x, train_y), "test": (test_x, test_y)}


def sample_ft_data(task_distribution, train_N=10):
    """Sample fine-tuning data from given task distribution."""
    if task_distribution == "amp_shift":
        amplitude = np.random.uniform(0.0, 2.0)
        bias = 0.0
    elif task_distribution == "bias_shift":
        amplitude = 1.0
        bias = np.random.uniform(-2, 2)
    elif task_distribution == "amp_bias_shift":
        amplitude = np.random.uniform(0.0, 3.0)
        bias = np.random.uniform(-2, 2)
    else:
        raise ValueError(f"Unknown task distribution {task_distribution}")
    ft_data = generate_sine_data(
        train_range=(0.0, 2.0),
        test_range=(0, 2 * np.pi),
        N=train_N,
        amplitude=amplitude,
        bias=bias,
    )
    train_x, train_y = ft_data["train"]
    test_x, test_y = ft_data["test"]
    train_x = torch.tensor(train_x).float().reshape(-1, 1)
    train_y = torch.tensor(train_y).float().reshape(-1, 1)
    test_x = torch.tensor(test_x).float().reshape(-1, 1)
    test_y = torch.tensor(test_y).float().reshape(-1, 1)
    return train_x, train_y, test_x, test_y


def savefig(fn):
    fig_path = Path("figures")
    fig_path.mkdir(exist_ok=True)
    plt.savefig(
        fig_path / fn, dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    pt_data = generate_sine_data()
    pt_x, pt_y = pt_data["train"]
    ft_data = generate_sine_data(
        train_range=(0, np.pi), test_range=(np.pi, 2 * np.pi), bias=-0.9
    )
    train_x, train_y = ft_data["train"]
    test_x, test_y = ft_data["test"]

    plt.figure(figsize=(4, 3))
    plt.scatter(pt_x, pt_y, edgecolors="k", c="gray", label="pretrain")
    plt.scatter(train_x, train_y, edgecolors="k", c="b", label="finetune")
    plt.scatter(test_x, test_y, edgecolors="k", c="r", label="test")
    plt.legend()
    plt.xlim(-0.2, 2 * np.pi + 0.2)
    plt.ylim(-2.0, 2.0)
    plt.title("Bias Shift Task")
    savefig("bias_shift_task.png")

    pt_data = generate_sine_data()
    pt_x, pt_y = pt_data["train"]
    ft_data = generate_sine_data(
        train_range=(0, np.pi), test_range=(np.pi, 2 * np.pi), amplitude=1.8
    )
    train_x, train_y = ft_data["train"]
    test_x, test_y = ft_data["test"]

    plt.figure(figsize=(4, 3))
    plt.scatter(pt_x, pt_y, edgecolors="k", c="gray", label="pretrain")
    plt.scatter(train_x, train_y, edgecolors="k", c="b", label="finetune")
    plt.scatter(test_x, test_y, edgecolors="k", c="r", label="test")
    plt.legend()
    plt.xlim(-0.2, 2 * np.pi + 0.2)
    plt.ylim(-2.0, 2.0)
    plt.title("Amplitude Shift Task")
    savefig("amplitude_shift_task.png")
