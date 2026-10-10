import argparse
import random
import torch

from cs336_basics.bpe_tokenizer import tokenizer
from cs336_basics.train_bpe import read_text_to_bpe
from cs336_basics.transformer import Transformer
from cs336_basics.train import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer text generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and BPE
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--bpe_dir", type=str, default="/data")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--model_context_length", type=int, default=256)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ffn_dim", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Checkpoint and device
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Generation hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--generation_context_length", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=100)

    return parser.parse_args()


args = parse_args()

# Model hyperparameters
vocab_size = args.vocab_size
model_context_length = args.model_context_length
model_dim = args.model_dim
num_layers = args.num_layers
num_heads = args.num_heads
ffn_dim = args.ffn_dim
rope_theta = args.rope_theta

# Device
device = args.device

# Generation hyperparameters
temperature = args.temperature
top_p = args.top_p
max_context_leng = args.generation_context_length
max_new_tokens = args.max_new_tokens

## 初始化模型 / 数据
vocab, merges = read_text_to_bpe(args.bpe_dir)
tkn = tokenizer(vocab, merges)
net = Transformer(vocab_size, model_context_length, model_dim,num_layers, num_heads, ffn_dim, rope_theta, device,)
inverse_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
net.load_state_dict(checkpoint["model"])
net.eval()

def main():
    while True:
        ## 输入prompt：
        generated_ids = []
        prompt = input(">>")
        # decode成 token_id [seq_len]
        inputs = tkn.encode(prompt)

        while True:
            if (prompt == "exit"):
                return
            model_inputs = inputs[-max_context_leng:].unsqueeze(0)
            # transformer 预测
            logits = net(model_inputs)
            # Temperature scaling softmax
            out = logits[len(model_inputs) - 1]
            prob = torch.exp(out / temperature) / torch.exp(out / temperature).sum(dim=-1, device=device)
            # Top-p 随机采样生成下一个token
            prob, index = torch.sort(prob, )
            # 找出remain项
            re_len, sum = 0, 0
            for p_t in prob:
                if sum >= top_p:
                    break
                sum += p_t
                re_len += 1

            # 重索引-计算概率
            re_prob, re_idx = prob[0:re_len], index[0:re_len]
            re_prob = re_prob / re_prob.sum()

            # 0 - 1 random_pro -> 随机采样
            random_t = random.random()
            sum, idx = 0, 0
            for i, b in enumerate(re_prob):
                sum += b
                if (sum >= random_t):
                    idx = i

            # 每次生成一个 token
            token_id_g = int(re_idx[idx].item())
            generated_ids.append(token_id_g)

            # 判断当前input维度：
            inputs.append(token_id_g)

            # 判断token 长度是否达到上限 / 是否生成结束 <|endoftext|>`
            if (
                inverse_vocab[b"<|endoftext|>"] == token_id_g
                or len(generated_ids) >= max_new_tokens
            ):
                break

        # 将所有 token 对应的 bytes 拼接后解码
        text = b"".join(vocab[i] for i in generated_ids).decode(
            "utf-8", errors="replace"
        )

        print(text)


if __name__ == "__main__":
    main()