Word = tuple[bytes, ...]

# Pair：词内一对相邻符号，是 BPE 统计和合并的对象
Pair = tuple[bytes, bytes]

import regex as re
from collections import OrderedDict
from collections.abc import Iterable, Iterator

class Tokenizer:

    """
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    """


    def __init__(self, vocab: dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        self.vocab, self.merges, self.special_token = vocab, merges, special_tokens

        self.merge_rank = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.reverse_vocab = {
            token: token_id for token_id, token in vocab.items()
        }
        # 初始化 BPE 缓存
        self._bpe_cache = OrderedDict()

        # 缓存最大容量
        self._bpe_cache_size = 100_000

        # 预先构造 256 个单字节 token
        self._byte_tokens = tuple(
            bytes([b]) for b in range(256)
        )

        # 初始化特殊 Token 集合
        self._special_tokens = set(special_tokens or [])

        # 编译特殊 Token 分割正则表达式
        if self._special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in sorted(
                    self._special_tokens,
                    key=len,
                    reverse=True
                )
            )

            self._special_pattern = re.compile(
                f"({special_pattern})"
            )
        else:
            self._special_pattern = None

        # 修复 6：编译预分词正则表达式
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self._pretoken_pattern = re.compile(PAT)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                token_id, token_hex = line.rstrip("\n").split("\t")

                vocab[int(token_id)] = bytes.fromhex(token_hex)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                token1_hex, token2_hex = line.rstrip("\n").split("\t")

                token1 = bytes.fromhex(token1_hex)
                token2 = bytes.fromhex(token2_hex)
                merges.append((token1, token2))
        if special_tokens:
            existing_tokens = set(vocab.values())
            for tt in special_tokens:
                token_bytes = tt.encode("utf-8")
                if token_bytes not in existing_tokens:
                    vocab[len(vocab)] = token_bytes
                    existing_tokens.add(token_bytes)
        return cls(vocab, merges, special_tokens)


    # def encode(self, text: str) -> list[int]:
    #     pre_tokens: list[list[bytes] | bytes] = []

    #     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    #     # special token 按长度降序放入捕获组，分割后 special token 本身仍会保留。
    #     if self.special_token:
    #         special_pattern = "|".join(
    #             re.escape(token)
    #             for token in sorted(set(self.special_token), key=len, reverse=True)
    #         )
    #         parts = re.split(f"({special_pattern})", text)
    #     else:
    #         parts = [text]

    #     # 1. pre-tokenize
    #     for part in parts:
    #         if not part:
    #             continue

    #         # special token 不参与普通预分词和 BPE，完整保存成 bytes。
    #         if self.special_token and part in self.special_token:
    #             pre_tokens.append(part.encode("utf-8"))
    #             continue

    #         for word in re.findall(PAT, part):
    #             word_bytes = word.encode("utf-8")
    #             word_list = list(
    #                 bytes([b]) for b in word_bytes
    #             )
    #             pre_tokens.append(word_list)

    #     result_tokens: list[bytes] = []

    #     # 2. 对每个 pre-token 执行 BPE
    #     for word in pre_tokens:

    #         # special token 已经是完整 bytes，直接按原位置放入最终 token 序列。
    #         if isinstance(word, bytes):
    #             result_tokens.append(word)
    #             continue

    #         symbols = word
    #         while len(symbols) >= 2:
    #             best_pair = None
    #             best_rank = float("inf")

    #             # 找当前优先级最高的 merge
    #             for i in range(len(symbols) - 1):
    #                 pair = (
    #                     symbols[i],
    #                     symbols[i + 1]
    #                 )
                    
    #                 if pair in self.merge_rank:
    #                     rank = self.merge_rank[pair]
    #                     if rank < best_rank:
    #                         best_rank = rank
    #                         best_pair = pair

    #             # 当前已经没有可以 merge 的 pair
    #             if best_pair is None:
    #                 break

    #             # 执行这个 merge
    #             new_symbols = []

    #             i = 0
    #             while i < len(symbols):
    #                 if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
    #                     new_symbols.append(best_pair[0] + best_pair[1])
    #                     i += 2          # 命中：两个符号换成新符号
    #                 else:
    #                     new_symbols.append(symbols[i])
    #                     i += 1

    #             symbols = new_symbols

    #         result_tokens.extend(symbols)

    #     tokenization_res: list[int] = []

    #     for token in result_tokens:
    #         tokenization_res.append(
    #             self.reverse_vocab[token]
    #         )

    #     return tokenization_res

    def encode(self, text: str) -> list[int]:
        pre_tokenization: list[int] = []
        merge_rank = self.merge_rank
        reverse_vocab = self.reverse_vocab
        cache = self._bpe_cache
        byte_tokens = self._byte_tokens

        parts = (
            re.splititer(self._special_pattern, text)
            if self._special_pattern is not None
            else (text,)
        )

        # 1. pre-tokenize
        for part in parts:
            if not part:
                continue

            # 直接保存 special token 
            if part in self._special_tokens:
                token_bytes = part.encode("utf-8")
                # special token 已经是完整 bytes，直接按原位置放入最终 token 序列。
                pre_tokenization.append(reverse_vocab[token_bytes])
                continue

            for match in self._pretoken_pattern.finditer(part):
                word_bytes = match.group().encode("utf-8")

                # 重复 pre-token 直接复用缓存，并更新最近使用顺序。
                cached_ids = cache.get(word_bytes)
                if cached_ids is not None:
                    cache.move_to_end(word_bytes)
                    pre_tokenization.extend(cached_ids)
                    continue

                # 2. 对每个 pre-token 执行 BPE
                symbols = [byte_tokens[b] for b in word_bytes]

                while len(symbols) >= 2:
                    best_pair = None
                    best_rank = float("inf")

                    # 找当前优先级最高的 merge
                    for i in range(len(symbols) - 1):
                        pair = (symbols[i], symbols[i + 1])
                        rank = merge_rank.get(pair)
                        if rank is not None and rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                    # 当前已经没有可以 merge 的 pair
                    if best_pair is None:
                        break

                    # 执行这个 merge
                    new_symbols = []
                    merged_symbol = best_pair[0] + best_pair[1]
                    i = 0
                    while i < len(symbols):
                        if (
                            i < len(symbols) - 1
                            and (symbols[i], symbols[i + 1]) == best_pair
                        ):
                            new_symbols.append(merged_symbol)
                            i += 2          # 命中：两个符号换成新符号
                        else:
                            new_symbols.append(symbols[i])
                            i += 1

                    symbols = new_symbols

                token_ids = tuple(reverse_vocab[token] for token in symbols)
                pre_tokenization.extend(token_ids)

                # 缓存容量达到上限时，移除最久未使用的 pre-token。
                if self._bpe_cache_size > 0:
                    cache[word_bytes] = token_ids
                    if len(cache) > self._bpe_cache_size:
                        cache.popitem(last=False)

        return pre_tokenization

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable:
            token_ids = self.encode(text)

            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
