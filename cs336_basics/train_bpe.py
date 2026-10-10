from pathlib import Path
from .bpe_tokenization import run_train_bpe1

# log code
from datetime import datetime
import logging
import time

# log code
logger = logging.getLogger(__name__)

def save_bpe_to_txt(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 vocab
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for token_id, token_bytes in vocab.items():
            f.write(f"{token_id}\t{token_bytes.hex()}\n")

    # 保存 merges
    with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            f.write(f"{token1.hex()}\t{token2.hex()}\n")

def read_text_to_bpe(
    input_dir: str
) ->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    input_dir = Path(input_dir)
    #
    with open(input_dir / "vocab.txt", "r", encoding="utf-8") as f:
        for line in f:
            ## 去掉换行符,按照\t切分
            token_id, byte = line.strip('\n').split('t')
            vocab[token_id] = bytes.fromhex(byte)
    # 保存 merges
    with open(input_dir / "merges.txt", "r", encoding="utf-8") as f:
        for line in f:
            byte1 ,byte2 = line.strip('\n').split('t')
            merges.append(bytes.fromhex(byte1), bytes.fromhex(byte2))

if __name__ == "__main__":
    # log code
    output_dir = Path("data/result")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    training_started = time.perf_counter()
    logger.info("训练程序启动 | log=%s", log_path)

    vocab, merges = run_train_bpe1(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    save_bpe_to_txt(
        vocab,
        merges,
        "data/result"
    )

    # log code
    logger.info(
        "训练结果已保存 | vocab=%s | merges=%s | total_elapsed=%.2fs",
        output_dir / "vocab.txt",
        output_dir / "merges.txt",
        time.perf_counter() - training_started,
    )
