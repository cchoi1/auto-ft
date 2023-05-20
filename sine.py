#%%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_sine_data(
    train_range=(0, 2 * np.pi),
    test_range=(0, 2 * np.pi),
    N=100,
    amplitude=1,
    phase=0,
    vertical_shift=0,
    noise_std=0.05,
):
    train_x = np.random.uniform(low=train_range[0], high=train_range[1], size=N)
    test_x = np.random.uniform(low=test_range[0], high=test_range[1], size=N)
    f = (
        lambda x: amplitude * np.sin(x + phase)
        + vertical_shift
        + np.random.normal(loc=0, scale=noise_std, size=x.shape)
    )
    train_y = f(train_x)
    test_y = f(test_x)
    return {"train": (train_x, train_y), "test": (test_x, test_y)}


if __name__ == "__main__":

    def savefig(fn):
        fig_path = Path("figures")
        fig_path.mkdir(exist_ok=True)
        plt.savefig(
            fig_path / fn, dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
        plt.show()
        plt.close()

    pt_data = generate_sine_data()
    pt_x, pt_y = pt_data["train"]
    ft_data = generate_sine_data(
        train_range=(0, np.pi), test_range=(np.pi, 2 * np.pi), vertical_shift=-0.9
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

    pt_data = sample_sine_data()
    pt_x, pt_y = pt_data["train"]
    ft_data = sample_sine_data(
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
# %%
