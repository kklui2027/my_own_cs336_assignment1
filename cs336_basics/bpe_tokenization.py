from collections import Counter
from pathlib import Path
from .pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool, cpu_count

# log code
import logging
import time

# log code
logger = logging.getLogger(__name__)

# Word：一个 pre-token 当前的符号序列，例：b"the" -> (b't', b'h', b'e')
# 键用 tuple 而不是 bytes，是为了保留"符号边界"：
# 迭代 bytes 得到的永远是单个字节，合并产生的多字节符号会被拆碎；
# tuple 里每个元素是一个独立符号，谁和谁相邻一目了然。
Word = tuple[bytes, ...]

# Pair：词内一对相邻符号，是 BPE 统计和合并的对象
Pair = tuple[bytes, bytes]


def merge_word(word: Word, pair: Pair, new_symbol: bytes) -> Word:
    """把 word 中所有（从左到右、不重叠）出现的 pair 替换成 new_symbol。"""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(new_symbol)
            i += 2          # 命中：两个符号换成新符号
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def para_pretoken(parts: list[str]) -> Counter[Word]:
    # 预分词
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counter: Counter[Word] = Counter()
    # 并行 对 每段普通文本按 PAT 预分词并计数。
    # findall 的文本参数必须是字符串，而 parts 是列表，
    # 所以要先 for part in parts 逐段处理。
    for word in parts:
        for word in re.findall(PAT, word):
            # findall 返回的是 str，训练全程基于 bytes，先编码
            word_bytes = word.encode("utf-8")
            # 迭代 bytes 得到的是 int（0~255），要重新包成单字节 bytes，
            # 得到这个 pre-token 的初始 Word：每个字节都是一个独立符号
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            token_counter[word_tuple] += 1
    return token_counter

def pretoken_chunk(
    task: tuple[str, int, int, str]
) -> Counter[Word]:
    """读取一个文件区间并执行预分词。"""
    input_path, start, end, special_pattern = task

    # 每个子进程独立打开只读文件，文件指针不会在进程之间互相影响。
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_pattern:
        parts = re.split(special_pattern, chunk)
    else:
        parts = [chunk]

    return para_pretoken(parts)

def run_train_bpe1(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]   
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # log code
    training_started = time.perf_counter()
    input_size = Path(input_path).stat().st_size
    logger.info(
        "开始 BPE 训练 | input=%s | size=%.2f MiB | vocab_size=%d | processes=%d",
        input_path,
        input_size / (1024 * 1024),
        vocab_size,
        cpu_count(),
    )

    ## 需要返回的数据， dict 词表，merges[]训练过程中合并的字符串
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    pre_token_counts: Counter[Word] = Counter()

    # Vocabulary initialization 全部byte的集合 0 ~ 255
    for i in range(0, 256, 1):
        vocab[i] = bytes([i])

    # 加入所有special_tokens
    for i, tt in enumerate(special_tokens):
        vocab[i + 256] = tt.encode("utf-8")


    # 特殊 token 的正则，按长度降序排列：防止短 token 是长 token 的前缀时把长 token 截断
    special_pattern = "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    )

    # log code
    pretoken_started = time.perf_counter()

    chunk_tasks: list[tuple[str, int, int, str]] = []
    with open(input_path, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_tasks.append((input_path, start, end, special_pattern))

    # log code
    logger.info(
        "预分词开始 | chunks=%d | processes=%d",
        len(chunk_tasks),
        cpu_count(),
    )

    with Pool(cpu_count()) as pool:
        # log code
        results = pool.imap_unordered(
            pretoken_chunk,
            chunk_tasks,
            chunksize=1,
        )

        for completed_chunks, result in enumerate(results, start=1):
            pre_token_counts.update(result)

            # log code
            logger.info(
                "预分词进度 | %d/%d chunks | unique_pre_tokens=%d | elapsed=%.2fs",
                completed_chunks,
                len(chunk_tasks),
                len(pre_token_counts),
                time.perf_counter() - pretoken_started,
            )

    # log code
    logger.info(
        "预分词完成 | token_occurrences=%d | unique_pre_tokens=%d | elapsed=%.2fs",
        sum(pre_token_counts.values()),
        len(pre_token_counts),
        time.perf_counter() - pretoken_started,
    )

    # ---- 第三步：BPE 训练主循环（增量更新版）----
    # pair_counts：全局 pair 加权频率
    # pair_to_words：倒排索引，pair -> 哪些词里有它，合并时用来直接定位受影响的词
    # 完整统计只做这一次，之后每轮合并只增量修正受影响的词。
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[Word]] = {}

    # log code
    pair_index_started = time.perf_counter()
    logger.info("开始构建 Pair 索引")

    for word, count in pre_token_counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] += count
            pair_to_words.setdefault(p, set()).add(word)

    # log code
    logger.info(
        "Pair 索引完成 | active_pairs=%d | elapsed=%.2fs",
        len(pair_counts),
        time.perf_counter() - pair_index_started,
    )
    merge_started = time.perf_counter()
    target_merges = max(0, vocab_size - len(vocab))
    log_every = 100
    logger.info("BPE 合并开始 | target_merges=%d", target_merges)

    while len(vocab) < vocab_size:
        if not pair_counts:
            # log code
            logger.warning("没有可继续合并的 pair，训练提前结束")
            break   # 没有任何相邻 pair 可合并了
        # 取最高频 pair；key 是 (计数, pair本身)，
        # 计数相同时元组比较落到 pair 上，自动取字典序更大的（作业要求）
        best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            # log code
            logger.warning("所有 pair 的计数均已耗尽，训练提前结束")
            break   # 所有 pair 都耗尽，提前结束
        new_symbol = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_symbol

        # 增量更新：只动含 best_pair 的词，其余词的 pair 计数原样有效
        for word in pair_to_words.pop(best_pair, set()):
            count = pre_token_counts.pop(word, None)
            if count is None:
                continue    # 过期索引：这个词在之前的合并里已经被改写掉了
            for i in range(len(word) - 1):          # 扣掉旧词的全部 pair 贡献
                pair_counts[(word[i], word[i + 1])] -= count
            new_word = merge_word(word, best_pair, new_symbol)
            for i in range(len(new_word) - 1):      # 加上新词的全部 pair 贡献
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += count
                pair_to_words.setdefault(p, set()).add(new_word)
            pre_token_counts[new_word] += count     # 记录这个词的当前形态

        # log code
        completed_merges = len(merges)
        if completed_merges == 1 or completed_merges % log_every == 0:
            merge_elapsed = time.perf_counter() - merge_started
            merge_rate = completed_merges / merge_elapsed if merge_elapsed > 0 else 0.0
            remaining_merges = target_merges - completed_merges
            eta_seconds = remaining_merges / merge_rate if merge_rate > 0 else 0.0
            progress = completed_merges / target_merges * 100 if target_merges else 100.0
            logger.info(
                "BPE 进度 | %d/%d merges (%.1f%%) | best_count=%d | active_pairs=%d "
                "| rate=%.2f merges/s | elapsed=%.1fs | eta=%.1fs",
                completed_merges,
                target_merges,
                progress,
                best_count,
                len(pair_counts),
                merge_rate,
                merge_elapsed,
                eta_seconds,
            )

    # log code
    logger.info(
        "BPE 训练完成 | vocab=%d | merges=%d | merge_elapsed=%.2fs | total_elapsed=%.2fs",
        len(vocab),
        len(merges),
        time.perf_counter() - merge_started,
        time.perf_counter() - training_started,
    )

    return vocab, merges
