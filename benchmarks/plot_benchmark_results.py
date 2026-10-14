# ai新增
import argparse
import csv
import json
import logging
import statistics
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
BLUE = "#3B6FB6"
ORANGE = "#C77C43"
GREEN = "#39836E"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    return rows


def format_axis(ax):
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.tick_params(direction="out", length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.04)


def plot_bpe_throughput(bpe_dir):
    bpe_dir = Path(bpe_dir)
    trials = read_csv(bpe_dir / "benchmark_trials.csv")
    config = json.loads((bpe_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    checksums = {item["token_ids_sha256"] for item in config["correctness"].values()}
    if len(checksums) != 1:
        raise ValueError("BPE 输出一致性检查未通过")

    variants = ["current", "optimized_no_cache", "optimized_lru_cold"]
    labels = ["Current", "Optimized\ncache off", "LRU\nempty start"]
    samples = [
        [float(row["tokens_per_second"]) / 1e6 for row in trials if row["variant"] == name]
        for name in variants
    ]
    medians = np.asarray([statistics.median(values) for values in samples])
    low = np.asarray([np.percentile(values, 25) for values in samples])
    high = np.asarray([np.percentile(values, 75) for values in samples])
    fig, ax = plt.subplots(figsize=(6.8, 4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.18)
    positions = np.arange(len(variants))
    ax.bar(
        positions, medians, width=0.58,
        color=[BLUE, "#929AA3", ORANGE],
        yerr=np.vstack([medians - low, high - medians]),
        error_kw={"ecolor": "#444444", "linewidth": 1, "capsize": 4},
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, high.max() * 1.16)
    ax.set_ylabel("Encoding throughput (M tokens/s)")
    ax.set_title(f"BPE throughput ({config['input_bytes'] / 1024**2:.0f} MiB, {config['repeats']} repeats)", pad=12)
    format_axis(ax)
    ax.plot([], [], color="#555555", linewidth=1, label="Median / IQR")
    ax.legend(loc="upper left")
    return fig


def read_kv_groups(kv_dirs):
    groups = {}
    configs = []
    for directory in kv_dirs:
        directory = Path(directory)
        config = json.loads((directory / "benchmark_config.json").read_text(encoding="utf-8"))
        if not config.get("cache_correctness"):
            raise ValueError(f"缺少 KV Cache 正确性记录：{directory}")
        summary = read_csv(directory / "benchmark_summary.csv")
        if any(row.get("same_sampled_ids_all_repeats") == "False" for row in summary):
            logging.warning("%s 中部分采样序列不同，详见原始日志", directory)
        configs.append(config)
        for row in read_csv(directory / "benchmark_trials.csv"):
            key = (row["mode"], int(row["prompt_tokens"]))
            if key not in groups:
                groups[key] = {"False": [], "True": [], "directory": str(directory), "new_tokens": int(row["new_tokens"])}
            group = groups[key]
            if group["directory"] != str(directory):
                raise ValueError(f"同一模式和前缀长度重复出现在两个目录：{key}")
            if group["new_tokens"] != int(row["new_tokens"]):
                raise ValueError("固定生成长度不一致")
            group[row["use_cache"]].append(float(row["seconds"]))
    signatures = {
        (c["checkpoint"], c["device"], c["new_tokens"], c["temperature"], c["top_p"])
        for c in configs
    }
    if len(signatures) != 1:
        raise ValueError("KV 目录中的模型、设备、输出长度或采样配置不同")
    for key, group in groups.items():
        if not group["False"] or len(group["False"]) != len(group["True"]):
            raise ValueError(f"缓存开关的测速次数不匹配：{key}")
    return groups, configs


def plot_kv_throughput(groups, configs, mode="end_to_end"):
    lengths = sorted(length for key_mode, length in groups if key_mode == mode)
    if not lengths:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.subplots_adjust(left=0.10, right=0.975, top=0.85, bottom=0.18, wspace=0.30)
    rate_ax, speed_ax = axes
    median_times = {}
    for enabled, label, color, line_style in [
        ("False", "No cache", BLUE, "-"),
        ("True", "KV cache", ORANGE, "--"),
    ]:
        median_times[enabled] = []
        rates, lows, highs = [], [], []
        for length in lengths:
            group = groups[mode, length]
            times = group[enabled]
            elapsed = statistics.median(times)
            median_times[enabled].append(elapsed)
            samples = group["new_tokens"] / np.asarray(times)
            rates.append(group["new_tokens"] / elapsed)
            lows.append(np.percentile(samples, 25))
            highs.append(np.percentile(samples, 75))
        rate_ax.plot(lengths, rates, color=color, linestyle=line_style,
                     linewidth=1.8, marker="o", markersize=4, label=label)
        rate_ax.fill_between(lengths, lows, highs, color=color, alpha=0.12, linewidth=0)

    speedups = np.asarray(median_times["False"]) / np.asarray(median_times["True"])
    speed_ax.axhline(1, color="#929AA3", linewidth=1, linestyle="--")
    speed_ax.plot(lengths, speedups, color=GREEN, linewidth=1.8,
                  marker="o", markersize=4)
    speed_ax.set_ylim(0.7, max(1.25, float(speedups.max()) * 1.18))
    rate_ax.set_ylabel("Median tokens/s")
    speed_ax.set_ylabel("Speedup over no cache")
    config = configs[0]
    title = "Generation" if mode == "end_to_end" else "Model-only"
    rate_ax.set_title(f"{title} throughput ({config['new_tokens']} new tokens)", pad=12)
    speed_ax.set_title("KV Cache speedup", pad=12)
    rate_ax.legend(loc="best", title="Shading: IQR", title_fontsize=9)
    rate_ax.set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_xticks(lengths)
        format_axis(ax)
    return fig


def plot_benchmark_results(bpe_dir, kv_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, configs = read_kv_groups(kv_dirs)
    with plt.rc_context(STYLE), PdfPages(output_dir / "benchmark_results.pdf") as pdf:
        builders = [
            ("bpe_throughput", lambda: plot_bpe_throughput(bpe_dir)),
            ("kv_cache_throughput", lambda: plot_kv_throughput(groups, configs, "end_to_end")),
            ("kv_cache_model_only", lambda: plot_kv_throughput(groups, configs, "model_only")),
        ]
        for name, build in builders:
            fig = build()
            if fig is None:
                continue
            try:
                fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white")
                pdf.savefig(fig, facecolor="white")
            finally:
                plt.close(fig)
    logging.info("测评图已保存到 %s", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe_dir", type=Path, required=True)
    parser.add_argument("--kv_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/benchmark_figures"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plot_benchmark_results(args.bpe_dir, args.kv_dir, args.output_dir)


if __name__ == "__main__":
    main()
