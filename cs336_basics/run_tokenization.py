import mmap
import multiprocessing as mp
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
from multiprocessing import cpu_count
import numpy as np
import time

_TOKENIZER = None
_SOURCE_FILE = None
_SOURCE = None


def _init_worker(raw_path, tkn):
    global _TOKENIZER, _SOURCE_FILE, _SOURCE

    _TOKENIZER = tkn

    if hasattr(tkn, "_bpe_cache"):
        tkn._bpe_cache.clear()

    _SOURCE_FILE = open(raw_path, "rb")
    _SOURCE = mmap.mmap(_SOURCE_FILE.fileno(), 0, access=mmap.ACCESS_READ)


# ai新增
def _iter_ranges(raw_path, chunk_size, boundary, start=0, end=None):
    # Generate byte ranges lazily, ending at complete document boundaries.
    with open(raw_path, "rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as text:
            size = len(text)
            if end is None:
                end = size

            while start < end:
                target = min(start + chunk_size, end)

                if target < end:
                    position = text.find(boundary, target, end)
                    range_end = position + len(boundary) if position != -1 else end
                else:
                    range_end = end

                yield start, range_end
                start = range_end


def _encode_range(start, end, part_path, boundary):
    # Decode and encode one complete document at a time.
    token_count = 0

    with open(part_path, "wb") as output:
        while start < end:
            position = _SOURCE.find(boundary, start, end)
            stop = position + len(boundary) if position != -1 else end

            text = _SOURCE[start:stop].decode("utf-8")
            token_ids = _TOKENIZER.encode(text)

            output.write(np.asarray(token_ids, dtype=np.uint32).tobytes())
            token_count += len(token_ids)

            start = stop

    return part_path, token_count


def _append_part(task, output):
    # Merge completed parts in source order.
    part_path, token_count = task.get()

    with open(part_path, "rb") as source:
        shutil.copyfileobj(source, output, length=1024 * 1024)

    Path(part_path).unlink()

    return token_count


# ai新增
def tokenize_file(
    raw_path,
    token_path,
    tkn,
    chunk_size=8 * 1024 * 1024,
    workers=None,
    start=0,
    end=None,
):
    raw_path = Path(raw_path).resolve()
    token_path = Path(token_path).resolve()

    boundary_text = "<|endoftext|>"
    boundary = boundary_text.encode("utf-8")
    special_tokens = set(tkn.special_token or [])

    if raw_path == token_path:
        raise ValueError("Input and output paths must differ.")

    if boundary_text not in special_tokens:
        raise ValueError("Configure <|endoftext|> as a special token.")

    if any(
        boundary_text in token and token != boundary_text
        for token in special_tokens
    ):
        raise ValueError("Other special tokens must not contain <|endoftext|>.")

    if workers is None:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )

    if workers < 1 or chunk_size < 1:
        raise ValueError("workers and chunk_size must be positive.")

    raw_size = raw_path.stat().st_size
    if end is None:
        end = raw_size
    if not 0 <= start <= end <= raw_size:
        raise ValueError("start and end must describe a valid byte range in the input.")

    token_path.parent.mkdir(parents=True, exist_ok=True)

    if start == end:
        token_path.write_bytes(b"")
        return 0

    if raw_size == 0:
        token_path.write_bytes(b"")
        return 0

    pending = deque()
    token_count = 0

    with tempfile.TemporaryDirectory(
        prefix="tokenize-",
        dir=token_path.parent,
    ) as temp_dir:
        temp_dir = Path(temp_dir)
        merged_path = temp_dir / "merged.bin"

        with mp.get_context("spawn").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(str(raw_path), tkn),
        ) as pool:
            with open(merged_path, "wb") as output:
                ranges = _iter_ranges(raw_path, chunk_size, boundary, start, end)

                for index, (range_start, range_end) in enumerate(ranges):
                    part_path = str(temp_dir / f"{index}.bin")

                    task = pool.apply_async(
                        _encode_range,
                        (range_start, range_end, part_path, boundary),
                    )

                    pending.append(task)

                    # Bound queued tasks and temporary output.
                    if len(pending) >= workers * 2:
                        token_count += _append_part(
                            pending.popleft(),
                            output,
                        )

                while pending:
                    token_count += _append_part(
                        pending.popleft(),
                        output,
                    )

        os.replace(merged_path, token_path)

    return token_count


# ai新增
def _find_split_offset(raw_path, validation_ratio, boundary):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    with open(raw_path, "rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as text:
            size = len(text)
            target = int(size * (1 - validation_ratio))
            offsets = []

            previous = text.rfind(boundary, 0, target)
            if previous != -1:
                offset = previous + len(boundary)
                if 0 < offset < size:
                    offsets.append(offset)

            following = text.find(boundary, target)
            if following != -1:
                offset = following + len(boundary)
                if 0 < offset < size:
                    offsets.append(offset)

            if not offsets:
                raise ValueError(
                    "Could not split the source into multiple documents at "
                    "<|endoftext|> boundaries."
                )

            return min(offsets, key=lambda offset: abs(offset - target))


# ai新增
def tokenize_train_val(
    raw_path,
    train_token_path,
    val_token_path,
    tkn,
    validation_ratio=0.1,
    chunk_size=8 * 1024 * 1024,
    workers=None,
):
    raw_path = Path(raw_path).resolve()
    train_token_path = Path(train_token_path).resolve()
    val_token_path = Path(val_token_path).resolve()

    if train_token_path == val_token_path:
        raise ValueError("Training and validation output paths must differ.")

    raw_size = raw_path.stat().st_size
    if raw_size == 0:
        train_count = tokenize_file(
            raw_path, train_token_path, tkn, chunk_size, workers, start=0, end=0,
        )
        val_count = tokenize_file(
            raw_path, val_token_path, tkn, chunk_size, workers, start=0, end=0,
        )
        return train_count, val_count

    boundary = b"<|endoftext|>"
    split_offset = _find_split_offset(raw_path, validation_ratio, boundary)

    train_count = tokenize_file(
        raw_path,
        train_token_path,
        tkn,
        chunk_size,
        workers,
        start=0,
        end=split_offset,
    )
    val_count = tokenize_file(
        raw_path,
        val_token_path,
        tkn,
        chunk_size,
        workers,
        start=split_offset,
        end=raw_size,
    )

    return train_count, val_count


from cs336_basics.bpe_tokenizer import Tokenizer

if __name__ == "__main__":

    tkn = Tokenizer.from_files("data/result/vocab.txt", "data/result/merges.txt", ["<|endoftext|>"])
    
    ## 统计token吞吐量： 
    # 保留 10% 文档作验证集。
    start_time = time.perf_counter()
    train_count, val_count = tokenize_train_val(
        raw_path="data/TinyStoriesV2-GPT4-train.txt",
        train_token_path="data/train.bin",
        val_token_path="data/val.bin",
        tkn=tkn,
        validation_ratio=0.1,
        chunk_size=8 * 1024 * 1024,
        workers=cpu_count(),
    )
    elapsed_seconds = time.perf_counter() - start_time 
    print(f"Token 吞吐量{(train_count + val_count) / elapsed_seconds:.4f}")
    print(f"Tokenization completed: train={train_count}, validation={val_count} tokens")
