# ai新增
import argparse
import ast
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def load_commented_encode(module):
    path = Path(module.__file__)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("    # def encode(")),
        None,
    )
    if start is None:
        raise ValueError(f"找不到已注释的缓存版 encode：{path}")
    recovered = []
    for line in lines[start:]:
        if line.startswith("    # "):
            recovered.append(line[6:])
        elif not line.strip():
            recovered.append("")
        else:
            break
    source = "\n".join(recovered).strip() + "\n"
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("缓存版代码必须只包含一个函数")
    namespace = dict(vars(module))
    exec(compile(tree, str(path) + ":commented_encode", "exec"), namespace)
    return namespace["encode"], source, start + 1


def read_documents(path, sample_bytes, boundary):
    with path.open("rb") as file:
        raw = file.read(sample_bytes + len(boundary))
    stop = raw.rfind(boundary)
    if stop < 0:
        raise ValueError("样本范围内没有文档分隔符，请增大 sample_mb")
    raw = raw[:stop + len(boundary)]
    marker = boundary.decode("utf-8")
    documents = [part.decode("utf-8") + marker for part in raw.split(boundary)[:-1]]
    return documents, raw


def encode_documents(tokenizer, documents, digest=False):
    import numpy as np

    count = 0
    checksum = hashlib.sha256() if digest else None
    for document in documents:
        ids = tokenizer.encode(document)
        count += len(ids)
        if checksum is not None:
            checksum.update(np.asarray(ids, dtype="<u4").tobytes())
    return count, checksum.hexdigest() if checksum is not None else None


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--bpe_dir", type=Path, default=Path("data/result"))
    parser.add_argument("--sample_mb", type=float, default=8)
    parser.add_argument("--cache_size", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/benchmark_bpe"))
    args = parser.parse_args()
    if args.sample_mb <= 0 or args.cache_size < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("sample_mb、cache_size、repeats 必须为正数，warmup >= 0")

    project_dir = args.project_dir.resolve()
    sys.path.insert(0, str(project_dir))
    import cs336_basics.bpe_tokenizer as module

    def resolve(path):
        return path if path.is_absolute() else project_dir / path

    input_path, bpe_dir = resolve(args.input), resolve(args.bpe_dir)
    documents, raw = read_documents(input_path, int(args.sample_mb * 1024**2), b"<|endoftext|>")
    cached_encode, cached_source, cached_line = load_commented_encode(module)
    variants = ["current", "optimized_no_cache", "optimized_lru_cold", "optimized_lru_warm"]
    tokenizers = {}
    for name in variants:
        tokenizer = module.Tokenizer.from_files(
            bpe_dir / "vocab.txt", bpe_dir / "merges.txt", ["<|endoftext|>"],
        )
        if name != "current":
            tokenizer.encode = cached_encode.__get__(tokenizer, type(tokenizer))
        tokenizer._bpe_cache_size = 0 if name == "optimized_no_cache" else args.cache_size
        tokenizers[name] = tokenizer

    reference = None
    checks = {}
    for name, tokenizer in tokenizers.items():
        tokenizer._bpe_cache.clear()
        count, digest = encode_documents(tokenizer, documents, digest=True)
        if reference is None:
            reference = (count, digest)
        if (count, digest) != reference:
            raise AssertionError(f"{name} 的 token ID 与当前实现不一致")
        checks[name] = {"tokens": count, "token_ids_sha256": digest}
        print(f"Correctness: {name}, tokens={count:,}, identical IDs", flush=True)

    def prepare(name):
        tokenizer = tokenizers[name]
        tokenizer._bpe_cache.clear()
        if name == "optimized_lru_warm":
            encode_documents(tokenizer, documents)

    trials = []
    times = {name: [] for name in variants}
    for _ in range(args.warmup):
        for name in variants:
            prepare(name)
            encode_documents(tokenizers[name], documents)

    for repeat in range(args.repeats):
        # 轮换顺序，避免所有基线都在同一温度或系统状态下运行。
        order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        for name in order:
            prepare(name)
            gc.collect()
            start = time.perf_counter()
            count, _ = encode_documents(tokenizers[name], documents)
            elapsed = time.perf_counter() - start
            if count != reference[0]:
                raise AssertionError("重复测速的 token 数不一致")
            times[name].append(elapsed)
            trials.append({
                "variant": name, "repeat": repeat + 1, "seconds": elapsed,
                "tokens": count, "tokens_per_second": count / elapsed,
                "mib_per_second": len(raw) / 1024**2 / elapsed,
                "cache_entries_after": len(tokenizers[name]._bpe_cache),
            })
            print(f"Repeat {repeat + 1}: {name}, {count / elapsed:,.0f} tokens/s", flush=True)

    medians = {name: statistics.median(values) for name, values in times.items()}
    rows = []
    for name in variants:
        elapsed = medians[name]
        rows.append({
            "variant": name, "documents": len(documents), "input_bytes": len(raw),
            "tokens": reference[0], "median_seconds": elapsed,
            "tokens_per_second": reference[0] / elapsed,
            "mib_per_second": len(raw) / 1024**2 / elapsed,
            "min_seconds": min(times[name]), "max_seconds": max(times[name]),
            "speedup_vs_current": medians["current"] / elapsed,
            "speedup_vs_optimized_no_cache": medians["optimized_no_cache"] / elapsed,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "benchmark_trials.csv", trials)
    write_csv(args.output_dir / "benchmark_summary.csv", rows)
    metadata = {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "input": str(input_path), "input_bytes": len(raw), "documents": len(documents),
        "input_sample_sha256": hashlib.sha256(raw).hexdigest(), "bpe_dir": str(bpe_dir),
        "vocab_sha256": hashlib.sha256((bpe_dir / "vocab.txt").read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256((bpe_dir / "merges.txt").read_bytes()).hexdigest(),
        "tokenizer_source": str(module.__file__),
        "tokenizer_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "commented_encode_start_line": cached_line,
        "recovered_encode_sha256": hashlib.sha256(cached_source.encode()).hexdigest(),
        "cache_size": args.cache_size, "repeats": args.repeats, "warmup": args.warmup,
        "correctness": checks,
        "scope": "Single-process encode only; input loading, tokenizer setup, uint32 conversion and file writing excluded.",
        "cold_cache": "Empty at the start of every measured full corpus pass; reused across documents within that pass.",
        "warm_cache": "Pre-encoded the same sample outside the timer; reported separately.",
        "source_handling": "Commented encode recovered in memory; project source files unchanged.",
    }
    (args.output_dir / "benchmark_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.output_dir / "measured_cached_encode.py").write_text(
        "# ai新增：从项目注释中提取的实际测速函数\n" + cached_source, encoding="utf-8",
    )
    print("\nMedian results:", flush=True)
    for row in rows:
        print(f"{row['variant']:24s} {row['tokens_per_second']:12,.0f} tokens/s, "
              f"vs current {row['speedup_vs_current']:.3f}x", flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
