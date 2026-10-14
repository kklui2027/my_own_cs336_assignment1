# ai新增
import argparse
import csv
import json
import time
from pathlib import Path


def show_bpe_results(result_dir):
    result_dir = Path(result_dir)
    config = json.loads((result_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    with (result_dir / "benchmark_summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    checks = config["correctness"]
    same_ids = len({item["token_ids_sha256"] for item in checks.values()}) == 1
    print("Saved BPE benchmark results\n")
    print(f"Input:   {Path(config['input']).name}")
    print(f"Sample:  {config['input_bytes'] / 1024**2:.2f} MiB | "
          f"{config['documents']:,} documents | {rows[0]['tokens'] and int(rows[0]['tokens']):,} tokens")
    print(f"Mode:    single process | {config['repeats']} repeats | "
          f"LRU capacity {config['cache_size']:,}")
    print(f"Token IDs: {'identical' if same_ids else 'MISMATCH'} (saved checks)\n")
    print(f"{'Variant':<27} {'Median (s)':>12} {'Tokens/s':>14} {'Speedup':>10}")
    print("-" * 68)
    labels = {
        "current": "Current",
        "optimized_no_cache": "Optimized / cache off",
        "optimized_lru_cold": "LRU / cold start",
        "optimized_lru_warm": "LRU / prewarmed",
    }
    for row in rows:
        print(f"{labels[row['variant']]:<27} {float(row['median_seconds']):>12.3f} "
              f"{float(row['tokens_per_second']):>14,.0f} "
              f"{float(row['speedup_vs_current']):>9.3f}x")
    print("\nCold start: cache cleared before each measured corpus pass.")
    print(f"Results: {result_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, default=Path("benchmarks/results/bpe"))
    parser.add_argument("--hold_seconds", type=int, default=0)
    args = parser.parse_args()
    show_bpe_results(args.result_dir)
    if args.hold_seconds > 0:
        time.sleep(min(args.hold_seconds, 60))


if __name__ == "__main__":
    main()
