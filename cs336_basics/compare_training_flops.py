# ai新增
import argparse
import csv
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FLOPS_PER_STEP = {
    "AttenRes": 916_337_983_488,
    "StdRes": 915_230_687_232,
}

STYLE = {
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


def read_metrics(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV 为空：{csv_path}")

    return {
        "steps": [int(row["step"]) for row in rows],
        "minutes": [float(row["elapsed_seconds"]) / 60 for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "val_loss": [float(row["val_loss"]) for row in rows],
        "learning_rate": [float(row["learning_rate"]) for row in rows],
    }


# ai新增
def get_loss_limits(runs, zoom_start_step):
    values = []

    for _, data, *_ in runs:
        indices = [
            index
            for index, step in enumerate(data["steps"])
            if step >= zoom_start_step
        ]

        # 训练步数较少时，使用后半段数据。
        if not indices:
            indices = range(len(data["steps"]) // 2, len(data["steps"]))

        for key in ("train_loss", "val_loss"):
            values.extend(
                data[key][index]
                for index in indices
                if math.isfinite(data[key][index])
            )

    if not values:
        raise ValueError("没有可用于设置纵轴范围的有限 loss")

    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.1, 0.02)

    return max(0.0, lower - padding), upper + padding


def compare_loss_curves(
    attenres_csv,
    stdres_csv,
    output_dir="res_comparison",
    loss_ylim=None,
    zoom_start_step=1000,
):
    output_dir = Path(output_dir)

    runs = [
        ("StdRes", read_metrics(stdres_csv), "-", "--", "-"),
        ("AttenRes", read_metrics(attenres_csv), ":", "-.", "--"),
    ]
    for name, data, *_ in runs:
        data["tflops"] = [
            step * FLOPS_PER_STEP[name] / 1e12
            for step in data["steps"]
        ]

    # ai新增
    if loss_ylim is None:
        loss_ylim = get_loss_limits(runs, zoom_start_step)

    if (
        not all(math.isfinite(value) for value in loss_ylim)
        or loss_ylim[0] >= loss_ylim[1]
    ):
        raise ValueError("loss_ylim 必须是两个有限数，且下限小于上限")

    columns = [
        ("steps", "Training step", "training step"),
        ("minutes", "Elapsed time (min)", "elapsed time"),
        (
            "tflops",
            "Cumulative training compute (TFLOPs)",
            "training compute",
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2, 3, figsize=(16.5, 7), constrained_layout=True,
        )

        try:
            for column, (x_key, xlabel, title) in enumerate(columns):
                loss_ax = axes[0, column]
                lr_ax = axes[1, column]

                for name, data, train_style, val_style, lr_style in runs:
                    x = data[x_key]

                    loss_ax.plot(
                        x, data["train_loss"],
                        color="#3B6FB6",
                        linewidth=1.8,
                        linestyle=train_style,
                        label=f"{name} / Train",
                    )
                    loss_ax.plot(
                        x, data["val_loss"],
                        color="#C77C43",
                        linewidth=1.8,
                        linestyle=val_style,
                        label=f"{name} / Validation",
                    )
                    lr_ax.plot(
                        x, data["learning_rate"],
                        color="#39836E",
                        linewidth=1.8,
                        linestyle=lr_style,
                        label=name,
                    )

                loss_ax.set_xlabel(xlabel)
                loss_ax.set_ylabel("Cross-entropy loss")
                loss_ax.set_title(f"Loss vs. {title}")
                loss_ax.legend(loc="best")

                # ai新增
                loss_ax.set_ylim(*loss_ylim)
                loss_ax.ticklabel_format(
                    axis="y", style="plain", useOffset=False,
                )

                lr_ax.set_xlabel(xlabel)
                lr_ax.set_ylabel("Learning rate")
                lr_ax.set_title(f"Learning rate vs. {title}")
                lr_ax.legend(loc="best")

                for ax in (loss_ax, lr_ax):
                    ax.grid(
                        axis="y", color="#E0E0E0", linewidth=0.6,
                    )
                    ax.tick_params(direction="out", length=3)
                    ax.margins(x=0.02)
                    ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=5, integer=column == 0)
                    )

                # ai新增
                loss_ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
                lr_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            for suffix in ("png", "pdf"):
                fig.savefig(
                    output_dir / f"training_comparison.{suffix}",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
        finally:
            plt.close(fig)

    logging.info("对比图已保存到 %s", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--AttenRes", dest="attenres_csv", type=Path, required=True,
    )
    parser.add_argument(
        "--StdRes", dest="stdres_csv", type=Path, required=True,
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("res_comparison"),
    )

    # ai新增
    parser.add_argument(
        "--loss_ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--zoom_start_step",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    logging.getLogger("fontTools").setLevel(logging.WARNING)

    compare_loss_curves(
        args.attenres_csv,
        args.stdres_csv,
        args.output_dir,
        loss_ylim=args.loss_ylim,
        zoom_start_step=args.zoom_start_step,
    )


if __name__ == "__main__":
    main()