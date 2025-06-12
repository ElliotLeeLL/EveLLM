from datetime import datetime

import matplotlib.pyplot as plt
from pathlib import Path


def plot_values(
        epochs_seen, examples_num, train_values, val_values,
        label="loss"
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="--", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_num, train_values, alpha=0)
    ax2.set_xlabel("Example seen")

    fig.tight_layout()
    plt.savefig(Path("result_diagrams") / f"{label}-plot-{datetime.now().strftime('%Y%m%d%H%M')}.pdf")
    plt.show()