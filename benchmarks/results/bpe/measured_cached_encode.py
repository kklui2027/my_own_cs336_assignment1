# ai新增：从项目注释中提取的实际测速函数
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
        if part is None:
            continue

        # 直接保存 special token 
        if part in self._special_tokens:
            token_bytes = part.encode("utf-8")
            # special token 已经是完整 bytes，直接按原位置放入最终 token 序列。
            pre_tokenization.append(reverse_vocab[token_bytes])
            continue

        for match in self._pretoken_pattern.finditer(part):
            word_bytes = match.group().encode("utf-8")

            # 重复 pre-token 直接复用缓存，执行LRU缓存策略
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
