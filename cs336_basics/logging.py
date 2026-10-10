# ai新增
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss_curves(run_dir):
    run_dir = Path(run_dir)

    with open(run_dir / "metrics.csv", encoding="utf-8", newline="") as file:
        data = [
            (
                int(row["step"]),
                float(row["elapsed_seconds"]) / 60,
                float(row["train_loss"]),
                float(row["val_loss"]),
            )
            for row in csv.DictReader(file)
        ]

    if not data:
        logging.warning("metrics.csv 为空，跳过绘图")
        return

    steps, minutes, train_losses, val_losses = zip(*data)

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }

    # 样式只作用于这张图。
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 3.5), constrained_layout=True
        )

        try:
            for ax, x, xlabel in [
                (axes[0], steps, "Training step"),
                (axes[1], minutes, "Elapsed time (min)"),
            ]:
                ax.plot(
                    x, train_losses,
                    color="#3B6FB6",
                    linewidth=1.8,
                    label="Train",
                )
                ax.plot(
                    x, val_losses,
                    color="#C77C43",
                    linewidth=1.8,
                    linestyle="--",
                    label="Validation",
                )

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Cross-entropy loss")
                ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
                ax.tick_params(direction="out", length=3)
                ax.margins(x=0.02)
                ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=5, integer=ax is axes[0])
                )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.legend(loc="best")

            for suffix in ("png", "pdf"):
                fig.savefig(
                    run_dir / f"loss_curves.{suffix}",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
        finally:
            plt.close(fig)

    logging.info("Loss curves saved to %s", run_dir)