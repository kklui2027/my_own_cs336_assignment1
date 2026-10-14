# ai新增
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch


def set_cache(model, enabled):
    model.reset_cache()
    for layer in model.layers:
        layer.mha.using_cache = enabled


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def check_cache(model, vocab_size, context_length, device):
    errors = {}
    prefix_lengths = sorted({min(16, context_length // 2), context_length // 2})
    for prefix_length in prefix_lengths:
        continuation_length = min(8, context_length - prefix_length)
        torch.manual_seed(123)
        tokens = torch.randint(
            vocab_size, (2, prefix_length + continuation_length), device=device,
        )
        set_cache(model, False)
        reference = model(tokens)

        for chunk_size in (1, 4):
            set_cache(model, True)
            prefix = model(tokens[:, :prefix_length])
            torch.testing.assert_close(
                prefix, reference[:, :prefix_length], atol=1e-4, rtol=1e-4,
            )
            outputs = []
            for start in range(prefix_length, tokens.shape[1], chunk_size):
                outputs.append(model(tokens[:, start:start + chunk_size]))
            actual = torch.cat(outputs, dim=1)
            expected = reference[:, prefix_length:]
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            errors[f"prefix_{prefix_length}_chunk_{chunk_size}_max_abs_error"] = float(
                (actual - expected).abs().max().cpu()
            )
            for layer in model.layers:
                assert layer.mha.past_k.shape[-2] == tokens.shape[1]
                assert layer.mha.past_v.shape[-2] == tokens.shape[1]

    set_cache(model, False)
    return errors


def sample_next(logits, temperature, top_p):
    # 沿用 decoding.py 的采样步骤。
    probabilities = torch.softmax(logits / temperature, dim=-1)
    probabilities, indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(probabilities, dim=-1)
    keep = cumulative - probabilities < top_p
    probabilities = probabilities[keep]
    indices = indices[keep]
    probabilities = probabilities / probabilities.sum()
    sample = torch.multinomial(probabilities, 1).item()
    return indices[sample].item()


@torch.inference_mode()
def measure_model_only(model, stream, prefix_length, new_tokens, use_cache, device):
    set_cache(model, use_cache)
    synchronize(device)
    start_time = time.perf_counter()
    first_token_seconds = None

    for index in range(new_tokens):
        if use_cache and index:
            inputs = stream[:, prefix_length + index - 1:prefix_length + index]
        else:
            inputs = stream[:, :prefix_length + index]
        model(inputs)
        if index == 0:
            synchronize(device)
            first_token_seconds = time.perf_counter() - start_time

    synchronize(device)
    return time.perf_counter() - start_time, first_token_seconds


@torch.inference_mode()
def measure_end_to_end(
    model, prompt, new_tokens, use_cache, device, temperature, top_p, seed,
):
    set_cache(model, use_cache)
    torch.manual_seed(seed)
    token_ids = list(prompt)
    generated = []
    synchronize(device)
    start_time = time.perf_counter()
    first_token_seconds = None

    for index in range(new_tokens):
        ids = token_ids[-1:] if use_cache and index else token_ids
        inputs = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(inputs)
        token = sample_next(logits[0, -1], temperature, top_p)
        token_ids.append(token)
        generated.append(token)
        if index == 0:
            synchronize(device)
            first_token_seconds = time.perf_counter() - start_time

    synchronize(device)
    elapsed = time.perf_counter() - start_time
    return elapsed, first_token_seconds, generated


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/step_9001.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--prompt")
    parser.add_argument("--token_file", type=Path)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--bpe_dir", type=Path, default=Path("data/result"))
    parser.add_argument("--new_tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes", nargs="+", choices=["model_only", "end_to_end"],
        default=["model_only", "end_to_end"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=Path, default=Path("kv_cache_benchmark"))
    args = parser.parse_args()

    if args.new_tokens < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error("new_tokens >= 2、repeats >= 1、warmup >= 0")
    if args.context_length < 4 or any(length < 1 for length in args.prompt_lengths):
        parser.error("context_length >= 4，prompt_lengths 必须为正数")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature 必须为正数，top_p 必须位于 (0, 1]")
    if args.prompt is not None and args.token_file is not None:
        parser.error("prompt 与 token_file 只能选择一个")
    if args.num_prompts < 1 or args.prompt_offset < 0:
        parser.error("num_prompts >= 1，prompt_offset >= 0")
    if args.prompt is None and any(
        length + args.new_tokens > args.context_length for length in args.prompt_lengths
    ):
        parser.error("prompt_lengths + new_tokens 不能超过 context_length")

    project_dir = args.project_dir.resolve()
    sys.path.insert(0, str(project_dir))
    from cs336_basics.transformer import Transformer

    device = torch.device(args.device)
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    vocab_size, d_model = state["emb.vocab"].shape
    layer_indices = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"layers\.(\d+)\.", key))
    }
    num_layers = max(layer_indices) + 1
    d_ff = state["layers.0.ffn.W1"].shape[0]
    using_attenres = any(re.fullmatch(r"layers\.\d+\.w[12]", key) for key in state)

    model = Transformer(
        vocab_size, args.context_length, d_model, num_layers, args.num_heads,
        d_ff, args.rope_theta,
        using_attenres=using_attenres, using_cache=False,
        device=device, dtype=torch.float32,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {'AttenRes' if using_attenres else 'StdRes'}, {num_layers} layers, d_model={d_model}")

    errors = check_cache(model, vocab_size, args.context_length, device)
    print(f"Cache correctness passed: {errors}")

    token_file_path = None
    continuation_cases = []
    case_ids = []
    if args.token_file is not None:
        import numpy as np

        token_file_path = args.token_file
        if not token_file_path.is_absolute():
            token_file_path = project_dir / token_file_path
        token_data = np.memmap(token_file_path, dtype=np.uint32, mode="r")
        prompt_cases = []
        for prompt_index in range(args.num_prompts):
            offset = args.prompt_offset + prompt_index * args.context_length
            for length in args.prompt_lengths:
                sample = token_data[offset:offset + length + args.new_tokens - 1]
                if len(sample) != length + args.new_tokens - 1:
                    parser.error("token_file 中没有足够的数据")
                if int(sample.max()) >= vocab_size:
                    parser.error("token_file 中的 ID 超过模型词表范围")
                prompt_cases.append(sample[:length].tolist())
                continuation_cases.append(sample[length:].tolist())
                case_ids.append(f"token_offset_{offset}")
    elif args.prompt is not None:
        from cs336_basics.bpe_tokenizer import Tokenizer
        from cs336_basics.train_bpe import read_text_to_bpe

        bpe_dir = args.bpe_dir
        if not bpe_dir.is_absolute():
            bpe_dir = project_dir / bpe_dir
        vocab, merges = read_text_to_bpe(str(bpe_dir))
        tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
        prompt_cases = [tokenizer.encode(args.prompt)]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        prompt_cases = [
            torch.randint(vocab_size, (length,), generator=generator).tolist()
            for length in args.prompt_lengths
        ]

    if not continuation_cases:
        continuation_cases = [None] * len(prompt_cases)
        case_ids = [f"prompt_{i}" for i in range(len(prompt_cases))]

    for prompt in prompt_cases:
        if not prompt or len(prompt) + args.new_tokens > args.context_length:
            parser.error("prompt 必须非空，且 prompt token 数 + new_tokens <= context_length")

    trial_rows = []
    summary_rows = []
    for prompt, fixed_continuation, case_id in zip(prompt_cases, continuation_cases, case_ids):
        prefix_length = len(prompt)
        torch.manual_seed(args.seed)
        continuation = (
            torch.tensor(fixed_continuation, dtype=torch.long, device=device)
            if fixed_continuation is not None
            else torch.randint(vocab_size, (args.new_tokens - 1,), device=device)
        )
        stream = torch.cat([
            torch.tensor(prompt, dtype=torch.long, device=device), continuation,
        ]).unsqueeze(0)

        for mode in args.modes:
            results = {False: [], True: []}
            generated_sequences = {False: [], True: []}

            def measure(enabled):
                if mode == "model_only":
                    elapsed, first = measure_model_only(
                        model, stream, prefix_length, args.new_tokens, enabled, device,
                    )
                    return elapsed, first, None
                return measure_end_to_end(
                    model, prompt, args.new_tokens, enabled, device,
                    args.temperature, args.top_p, args.seed,
                )

            for _ in range(args.warmup):
                for enabled in (False, True):
                    measure(enabled)

            for repeat in range(args.repeats):
                order = (False, True) if repeat % 2 == 0 else (True, False)
                for enabled in order:
                    elapsed, first, generated = measure(enabled)
                    results[enabled].append((elapsed, first))
                    generated_sequences[enabled].append(generated)
                    trial_rows.append({
                        "case_id": case_id,
                        "mode": mode,
                        "prompt_tokens": prefix_length,
                        "new_tokens": args.new_tokens,
                        "use_cache": enabled,
                        "repeat": repeat + 1,
                        "seconds": elapsed,
                        "first_token_seconds": first,
                        "tokens_per_second": args.new_tokens / elapsed,
                    })

            medians = {
                enabled: statistics.median(elapsed for elapsed, _ in samples)
                for enabled, samples in results.items()
            }
            speedup = medians[False] / medians[True]
            for enabled in (False, True):
                elapsed = medians[enabled]
                decode_seconds = statistics.median(
                    total - first for total, first in results[enabled]
                )
                summary_rows.append({
                    "case_id": case_id,
                    "mode": mode,
                    "prompt_tokens": prefix_length,
                    "new_tokens": args.new_tokens,
                    "use_cache": enabled,
                    "median_seconds": elapsed,
                    "tokens_per_second": args.new_tokens / elapsed,
                    "decode_tokens_per_second": (args.new_tokens - 1) / decode_seconds,
                    "min_seconds": min(total for total, _ in results[enabled]),
                    "max_seconds": max(total for total, _ in results[enabled]),
                    "cache_speedup": speedup,
                    "same_sampled_ids_all_repeats": (
                        all(a == b for a, b in zip(
                            generated_sequences[False], generated_sequences[True],
                        )) if mode == "end_to_end" else "not_applicable"
                    ),
                })
            print(
                f"{mode:12s} {case_id} prompt={prefix_length:3d}, new={args.new_tokens}: "
                f"no_cache={args.new_tokens / medians[False]:.2f} token/s, "
                f"cache={args.new_tokens / medians[True]:.2f} token/s, "
                f"speedup={speedup:.3f}x",
                flush=True,
            )
            if mode == "end_to_end":
                print(
                    "  same sampled token IDs:",
                    all(a == b for a, b in zip(
                        generated_sequences[False], generated_sequences[True],
                    )),
                    flush=True,
                )

    set_cache(model, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "benchmark_trials.csv", trial_rows)
    write_csv(args.output_dir / "benchmark_summary.csv", summary_rows)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(next(model.parameters()).dtype),
        "python_version": sys.version,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "using_attenres": using_attenres,
        "context_length": args.context_length,
        "prompt_lengths": [len(prompt) for prompt in prompt_cases],
        "token_file": str(token_file_path) if token_file_path is not None else None,
        "prompt_offset": args.prompt_offset,
        "num_prompts": args.num_prompts,
        "prompt_tokens_sha256": [
            hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            for prompt in prompt_cases
        ],
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "modes": args.modes,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "eos_stop_disabled_for_fixed_length_benchmark": True,
        "cache_correctness": errors,
        "measurement_scope": "Model-only uses identical teacher-forced token streams; end-to-end includes prefill, tensor construction, transfer and sampling, but excludes tokenizer loading, text encoding and final text decoding.",
        "rolling_window": "Disabled for both variants; prompt length + new tokens is bounded by context_length.",
        "model_source_sha256": {
            name: hashlib.sha256((project_dir / "cs336_basics" / name).read_bytes()).hexdigest()
            for name in ("model.py", "transformer.py", "decoding.py")
        },
    }
    (args.output_dir / "benchmark_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
