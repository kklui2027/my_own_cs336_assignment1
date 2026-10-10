from cs336_basics.train_bpe import read_txt_to_bpe
from cs336_basics.transformer import Transformer
from cs336_basics.train import AdamW, cross_entropy, learning_rate_schedule,  gradient_clipping, data_loading, save_checkpoint
from pathlib import Path
import numpy as np
import argparse
import torch


## 读取命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer 语言模型训练参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## 数据与 BPE
    data_args = parser.add_argument_group("数据与 BPE")
    data_args.add_argument("--train_path", type=str, required=True)
    data_args.add_argument("--val_path", type=str, required=True)
    data_args.add_argument("--bpe_dir", type=str, default=None)

    ## Transformer 模型
    model_args = parser.add_argument_group("Transformer 模型")
    model_args.add_argument("--vocab_size", type=int, default=10000)
    model_args.add_argument("--context_length", type=int, default=256)
    model_args.add_argument("--d_model", type=int, default=512)
    model_args.add_argument("--num_layers", type=int, default=4)
    model_args.add_argument("--num_heads", type=int, default=16)
    model_args.add_argument("--d_ff", type=int, default=1344)
    model_args.add_argument("--rope_theta", type=float, default=10000.0)

    ## AdamW 优化器
    optimizer_args = parser.add_argument_group("AdamW 优化器")
    optimizer_args.add_argument("--lr", type=float, default=3e-4)
    optimizer_args.add_argument("--beta1", type=float, default=0.9)
    optimizer_args.add_argument("--beta2", type=float, default=0.999)
    optimizer_args.add_argument("--eps", type=float, default=1e-8)
    optimizer_args.add_argument("--weight_decay", type=float, default=0.01)

    ## 学习率调度
    schedule_args = parser.add_argument_group("学习率调度")
    schedule_args.add_argument("--max_learning_rate", type=float, default=None)
    schedule_args.add_argument("--min_learning_rate", type=float, default=None)
    schedule_args.add_argument("--warmup_iters", type=int, default=100)
    schedule_args.add_argument("--cosine_cycle_iters", type=int, default=None)

    ## 训练与运行设备
    training_args = parser.add_argument_group("训练与运行设备")
    training_args.add_argument("--batch_size", type=int, default=32)
    training_args.add_argument("--max_steps", type=int, default=10000)
    training_args.add_argument("--device", type=str, default="cuda")
    training_args.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")

    ## 梯度裁剪
    clipping_args = parser.add_argument_group("梯度裁剪")
    clipping_args.add_argument("--max_l2_norm", type=float, default=1.0)

    ## 验证
    evaluation_args = parser.add_argument_group("验证")
    evaluation_args.add_argument("--eval_interval", type=int, default=100)
    evaluation_args.add_argument("--eval_batches", type=int, default=10)

    ## 检查点保存与恢复
    checkpoint_args = parser.add_argument_group("检查点保存与恢复")
    checkpoint_args.add_argument("--save_interval", type=int, default=1000)
    checkpoint_args.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    checkpoint_args.add_argument("--resume_path", type=str, default=None)

    return parser.parse_args()

def main():
    # 读命令行参数：
    args = parse_args()

    ## 数据与 BPE
    train_path = args.train_path
    val_path = args.val_path
    bpe_dir = args.bpe_dir

    ## Transformer 模型
    vocab_size = args.vocab_size
    context_length = args.context_length
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = args.rope_theta

    ## AdamW 优化器
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    eps = args.eps
    weight_decay = args.weight_decay

    ## 学习率调度
    max_learning_rate = args.max_learning_rate if args.max_learning_rate is not None else lr
    min_learning_rate = args.min_learning_rate if args.min_learning_rate is not None else 0.1 * max_learning_rate
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.max_steps

    ## 训练与运行设备
    batch_size = args.batch_size
    max_steps = args.max_steps
    device = args.device
    dtype = getattr(torch, args.dtype)

    ## 梯度裁剪
    max_l2_norm = args.max_l2_norm

    ## 验证
    eval_interval = args.eval_interval
    eval_batches = args.eval_batches

    ## 检查点保存与恢复
    save_interval = args.save_interval
    checkpoint_dir = args.checkpoint_dir
    resume_path = args.resume_path

    ##  从训练好的merge / vocab 中加载数据
    # data_loading 随机读取训练数据
    train_data = np.memmap()
    valid_data = np.memmap()
    ## 初始化环境
    # 核心网络
    net = Transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device, dtype=dtype,)
    # 损失函数
    loss_fn = cross_entropy
    # optimizer 优化器
    optimizer = AdamW(net.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,)
    # 学习率调度器
    scheduler = learning_rate_schedule
    # 梯度剪裁
    gradient_regularization = gradient_clipping

    ## 训练循环
    for step in range(max_steps):
        ## 调整学习率：
        lr = scheduler(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ## 训练 -> 反向传播 -> 梯度剪裁 -> 优化
        net.train()
        net.zero_grad()
        inputs, labels = data_loading(train_path, batch_size, context_length, device)
        logits = net(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        gradient_regularization(net.parameters(), max_l2_norm)
        optimizer.step()

        if (step % eval_interval == 0):
            with torch.no_grad():
                train_loss_sum, valid_loss_sum = 0, 0
                for st in range(eval_batches):
                    train_inputs, train_labels = data_loading(train_path, batch_size, context_length, device)
                    valid_inputs, valid_labels= data_loading(val_path, batch_size, context_length, device)
                    train_logits = net(train_inputs)
                    valid_logits = net(valid_inputs)
                    valid_loss = loss_fn(valid_logits, valid_labels)
                    train_loss = loss_fn(train_logits, train_labels)
                    train_loss_sum += train_loss
                    valid_loss_sum += valid_loss
                print(f"Step.{step + 1} train_loss={train_loss_sum / eval_batches :.4f} valid={valid_loss_sum / eval_batches:.4f} lr={lr:.6g}")
                    

        if (step % save_interval == 0):
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            save_path = checkpoint_path / f"step_{step + 1}.pt"

            save_checkpoint(net, optimizer, step, save_path)
            print(f"Step.{step + 1} already check point in {save_path}")



if __name__ == "__main__":
    main()