# ai新增
import argparse
import csv
import logging
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


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

COLORS = {"StdRes": "#3B6FB6", "AttenRes": "#C77C43"}


def read_norms(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        required = {"step", "weight_l2", "grad_l2_pre_clip"}
        missing = required - columns
        if missing:
            raise ValueError(f"{csv_path} 缺少字段：{', '.join(sorted(missing))}")

        samples = {}
        for row in reader:
            step = int(row["step"])
            if step in samples:
                raise ValueError(f"{csv_path} 存在重复 step：{step}")
            samples[step] = row

    if not samples:
        raise ValueError(f"CSV 为空：{csv_path}")

    return samples, columns


def _layer_metrics(columns, suffix):
    pattern = re.compile(rf"layer_(\d+)_{re.escape(suffix)}")
    metrics = []
    for column in columns:
        match = pattern.fullmatch(column)
        if match:
            layer = int(match.group(1))
            metrics.append((layer, column, f"Layer {layer}"))
    return [(column, title) for _, column, title in sorted(metrics)]


def _values(samples, steps, metric):
    return np.asarray(
        [
            float(samples[step][metric])
            if samples[step][metric].strip()
            else np.nan
            for step in steps
        ],
        dtype=np.float64,
    )


def _summary(metric, model, values):
    finite_values = values[np.isfinite(values)]
    result = {
        "metric": metric,
        "model": model,
        "count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "zero_count": int((finite_values == 0).sum()),
    }
    result.update({key: "" for key in ("mean", "std", "median", "min", "max")})
    if len(finite_values):
        result.update({
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
            "median": float(np.median(finite_values)),
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
        })
    return result


def _plot_group(models, steps, metrics, ylabel, title):
    columns = 2 if len(metrics) <= 4 else 3
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=12)
    rng = np.random.default_rng(0)
    summaries = []

    try:
        for ax, (metric, panel_title) in zip(axes.flat, metrics):
            nonfinite_counts = []

            for position, (name, samples) in enumerate(models.items(), start=1):
                values = _values(samples, steps, metric)
                summaries.append(_summary(metric, name, values))
                finite_values = values[np.isfinite(values)]
                nonfinite_count = int((~np.isfinite(values)).sum())
                nonfinite_counts.append((name, nonfinite_count))
                color = COLORS[name]

                if len(finite_values):
                    ax.boxplot(
                        [finite_values],
                        positions=[position],
                        widths=0.45,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        boxprops={
                            "facecolor": matplotlib.colors.to_rgba(color, 0.16),
                            "edgecolor": color,
                            "linewidth": 1.2,
                        },
                        medianprops={"color": color, "linewidth": 1.8},
                        whiskerprops={"color": color, "linewidth": 1.2},
                        capprops={"color": color, "linewidth": 1.2},
                    )
                    jitter = rng.uniform(-0.07, 0.07, len(finite_values))
                    ax.scatter(
                        position + jitter,
                        finite_values,
                        s=12,
                        color=color,
                        alpha=0.3,
                        edgecolors="none",
                        zorder=2,
                    )
                else:
                    ax.text(
                        position, 0.5, "No finite data",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="center", color=color, fontsize=9,
                    )

                if nonfinite_count:
                    logging.warning(
                        "%s 的 %s 含有 %s 个 NaN/Inf 或空值",
                        name, metric, nonfinite_count,
                    )

            if any(count for _, count in nonfinite_counts):
                message = "NaN/Inf or missing: " + ", ".join(
                    f"{name}={count}" for name, count in nonfinite_counts
                )
                ax.text(
                    0.02, 0.97, message,
                    transform=ax.transAxes, va="top", fontsize=8,
                )

            ax.set_xticks([1, 2], ["StdRes", "AttenRes"])
            ax.set_xlim(0.5, 2.5)
            ax.margins(y=0.08)
            ax.set_ylim(bottom=0)
            ax.set_title(panel_title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
            ax.tick_params(direction="out", length=3)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)

        return fig, summaries
    except Exception:
        plt.close(fig)
        raise


def compare_norms(
    attenres_csv,
    stdres_csv,
    output_dir="norms_comparison",
    start_step=None,
    end_step=None,
):
    stdres, stdres_columns = read_norms(stdres_csv)
    attenres, attenres_columns = read_norms(attenres_csv)
    models = {"StdRes": stdres, "AttenRes": attenres}

    # 用相同的训练步做对比。
    steps = sorted(stdres.keys() & attenres.keys())
    steps = [
        step for step in steps
        if (start_step is None or step >= start_step)
        and (end_step is None or step <= end_step)
    ]
    if not steps:
        raise ValueError("所选区间内没有两个模型共有的 step")

    columns = stdres_columns & attenres_columns
    activations = _layer_metrics(columns, "activation_rms")
    if not activations:
        raise ValueError("两个 CSV 没有共有的 layer_*_activation_rms 字段")

    groups = [
        (
            "activation_norms_comparison", "Activation norms", "Activation RMS",
            activations,
        ),
        (
            "gradient_norms_comparison", "Gradient norms",
            "Gradient L2 norm (pre-clip)",
            [("grad_l2_pre_clip", "All parameters")]
            + _layer_metrics(columns, "grad_l2_pre_clip"),
        ),
        (
            "weight_norms_comparison", "Weight norms", "Weight L2 norm",
            [("weight_l2", "All parameters")]
            + _layer_metrics(columns, "weight_l2"),
        ),
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with plt.rc_context(STYLE):
        with PdfPages(output_dir / "norms_comparison.pdf") as pdf:
            for filename, title, ylabel, metrics in groups:
                fig, group_summaries = _plot_group(
                    models, steps, metrics, ylabel, title,
                )
                try:
                    fig.savefig(
                        output_dir / f"{filename}.png",
                        dpi=300,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                    summaries.extend(group_summaries)
                finally:
                    plt.close(fig)

    with (output_dir / "norms_summary.csv").open(
        "w", encoding="utf-8", newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    logging.info(
        "比较 %s 个共有 step（%s—%s），图表已保存到 %s",
        len(steps), steps[0], steps[-1], output_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--AttenRes", dest="attenres_csv", type=Path, required=True)
    parser.add_argument("--StdRes", dest="stdres_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("norms_comparison"))
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--end_step", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    compare_norms(
        args.attenres_csv,
        args.stdres_csv,
        args.output_dir,
        args.start_step,
        args.end_step,
    )


if __name__ == "__main__":
    main()
