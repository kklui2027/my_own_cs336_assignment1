# ai新增
import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss_curves(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.csv"

    with metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        logging.warning("metrics.csv 为空，跳过绘图")
        return

    steps = [int(row["step"]) for row in rows]
    minutes = [float(row["elapsed_seconds"]) / 60 for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]
    learning_rates = [float(row["learning_rate"]) for row in rows]

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

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11, 7),
            constrained_layout=True,
        )

        try:
            # Loss 随训练步数变化
            axes[0, 0].plot(
                steps,
                train_losses,
                color="#3B6FB6",
                linewidth=1.8,
                label="Train",
            )
            axes[0, 0].plot(
                steps,
                val_losses,
                color="#C77C43",
                linewidth=1.8,
                linestyle="--",
                label="Validation",
            )
            axes[0, 0].set_xlabel("Training step")
            axes[0, 0].set_ylabel("Cross-entropy loss")
            axes[0, 0].set_title("Loss vs. training step")
            axes[0, 0].legend(loc="best")

            # Loss 随耗时变化
            axes[0, 1].plot(
                minutes,
                train_losses,
                color="#3B6FB6",
                linewidth=1.8,
                label="Train",
            )
            axes[0, 1].plot(
                minutes,
                val_losses,
                color="#C77C43",
                linewidth=1.8,
                linestyle="--",
                label="Validation",
            )
            axes[0, 1].set_xlabel("Elapsed time (min)")
            axes[0, 1].set_ylabel("Cross-entropy loss")
            axes[0, 1].set_title("Loss vs. elapsed time")
            axes[0, 1].legend(loc="best")

            # Learning rate 随训练步数变化
            axes[1, 0].plot(
                steps,
                learning_rates,
                color="#39836E",
                linewidth=1.8,
            )
            axes[1, 0].set_xlabel("Training step")
            axes[1, 0].set_ylabel("Learning rate")
            axes[1, 0].set_title("Learning rate vs. training step")

            # Learning rate 随耗时变化
            axes[1, 1].plot(
                minutes,
                learning_rates,
                color="#39836E",
                linewidth=1.8,
            )
            axes[1, 1].set_xlabel("Elapsed time (min)")
            axes[1, 1].set_ylabel("Learning rate")
            axes[1, 1].set_title("Learning rate vs. elapsed time")

            for ax in axes.flat:
                ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
                ax.tick_params(direction="out", length=3)
                ax.margins(x=0.02)
                ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=5, integer=ax in (axes[0, 0], axes[1, 0]))
                )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            # 保存的是同一张包含四个子图的图，因此 PDF 和 PNG 都有全部内容。
            fig.savefig(
                run_dir / "training_curves.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                run_dir / "training_curves.pdf",
                bbox_inches="tight",
                facecolor="white",
            )
        finally:
            plt.close(fig)

    logging.info("训练曲线已保存到 %s", run_dir)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="包含 metrics.csv 的训练结果目录",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    plot_loss_curves(args.run_dir)


if __name__ == "__main__":
    main()