# CS336 语言模型实现与实验

这个仓库记录了我基于 [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) 实现一个小型 Decoder-Only Transformer 的过程。我用 PyTorch 的张量运算和自动微分搭建模型，在 TinyStories 上完成训练和文本生成，之后又尝试了 BPE 编码缓存、KV Cache，以及跨层残差注意力。

通过这个项目我把分词、训练和推理连成线：从文本如何变成 token ID，前向传播中每一步张量维度的变化，到每个位置的损失的计算，再到模型逐个生成词元。仓库里保留了所有实验的记录，包括有收益的算法优化，以及架构的修改尝试。

## 实现内容

- **分词与数据处理**：字节级 BPE、词表与合并规则的保存和加载、特殊词元处理、LRU 编码缓存、多进程分块编码、训练与验证数据划分。
- **模型**：自定义 Linear、Embedding、RMSNorm、因果多头自注意力、RoPE、SwiGLU，以及 Transformer Block 和语言模型输出层。
- **训练**：交叉熵、AdamW、学习率预热与余弦衰减、梯度裁剪、检查点保存与加载、训练和验证循环。
- **生成**：自回归解码、温度缩放、top-p 采样和逐层 KV Cache。
- **实验**：学习率对比、StdRes 与 AttenRes 对照、激活和梯度范数记录、算子 FLOPs 统计、分词与生成吞吐量测评。

主要代码位置：

```text
cs336_basics/
├── bpe_tokenization.py         BPE 训练实现
├── train_bpe.py                BPE 训练入口、词表与合并规则读写
├── bpe_tokenizer.py            文本编码与解码、LRU 缓存
├── run_tokenization.py         多进程编码、数据划分与二进制输出
├── model.py                    Linear、RMSNorm、RoPE、SwiGLU、MHA 和 KV Cache
├── transformer.py              Transformer Block、语言模型、AttenRes
├── train.py                    损失、优化器、调度、裁剪、采样数据和检查点
├── run_train.py                训练、验证与实验日志
├── decoding.py                 文本生成入口
├── logging.py                  训练曲线绘图
├── floats_evaluate.py          算子 FLOPs 测评
├── compare_training_flops.py   损失、耗时与累计计算量对比
└── compare_norms.py            激活、梯度和权重范数对比

benchmarks/
├── benchmark_tokenizer.py      BPE 编码对照测评
├── benchmark_kv_cache.py       KV Cache 正确性与吞吐量测评
├── plot_benchmark_results.py   测评结果绘图
└── results/                    原始计时、配置和结果汇总

tests/
└── adapters.py                与课程测试接口对接
```

下面的实验使用 4 层、`d_model=512`、16 个注意力头、`d_ff=1344`、词表大小 10,000 的模型，训练上下文长度为 256。模型规模较小，主要用来验证实现和观察不同设计的行为。

## 实现中遇到的困难

### 数学里的列向量和代码里的张量布局

多头注意力最开始容易弄错的地方，是投影矩阵应该存成什么形状。尤其 Q、K、V 的矩阵都是方阵，转置写反了也不会立刻报维度错误。

数学里可以写成 `y = Wx`，但 PyTorch 通常把特征放在最后一维。权重采用 `(out_features, in_features)` 的布局时，对应的实现是：

```python
y = x @ weight.T
```

Q、K、V 投影完成后，张量从 `(B, T, D)` 拆成 `(B, H, T, d_k)`；注意力计算完成后再合并回 `(B, T, D)`。其维度含义需要与适配器传入的权重排列一致。

### 交叉熵的目标是 token ID

语言模型的输出是 `(B, T, V)` 的 logits，目标是 `(B, T)` 的下一个 token ID。数据采样时，输入和标签来自同一段文本，但标签向后移动一位：

```text
输入：x0, x1, x2, ..., x(T-1)
目标：x1, x2, x3, ..., xT
```

每个位置都是一次下一词元分类，最后对 `B × T` 个位置的损失取平均，得到用于反向传播的标量。目标类别决定取哪个 logit，但归一化项仍然依赖整个词表的 logits。

另一个问题是数值稳定性。先算 softmax 再取 log，在Float32中概率值可能会下溢到 0，通过数学变形：

$$
\ell(z,y)=\log\sum_j\exp(z_j-m)-(z_y-m),\qquad m=\max_j z_j
$$

减去了最大值，避免了对已下溢的概率取对数。模型返回原始 logits，softmax 留到采样阶段使用。

### KV Cache 不只是保存两个张量

我在加入缓存时，需要重新确认每一步到底应该计算哪些位置。首轮处理完整前缀，之后只输入新产生单个 token；Q、K、V 都只针对新增位置计算，再把新的 K、V 沿序列维度接到各层自己的缓存后面。

实现过程中的细节：

- **RoPE 的位置偏移**：新 token 的位置从已有缓存长度开始，不能每次都从 0 编号。
- **K 的保存时机**：缓存已经应用 RoPE 的 K，避免下次把历史 K 再旋转一次；V 直接保存。
- **因果掩码**：增量输入的 Query 长度和完整 Key 长度不同。新 Query 的第 `i` 个位置只能读取 `j <= past_len + i` 的 Key。
- **缓存生命周期**：不同 prompt 之间、切换缓存模式之前，都需要清空各层缓存。
- **开关**：解码输入与注意力层的缓存需要保持一致。只让解码器输入最后一个 token，却没有让模型保存历史 K/V，会丢失上下文。

## 优化

## BPE 编码优化

分词过程中存在反复处理出现过的预分词片段现象。逐次编码时，对这些片段重新执行 BPE 合并会产生很多重复运算。我用片段的原始 bytes 作为Key，将其对应的 token ID 存储下来，用LRU策略控制缓存容量。

我比较了当前版本从空缓存开始的 LRU 版本。样本是 TinyStories 源文本约 8 MiB，2,037,821 个 token，各版本测量 5 次。

| 编码版本  | 吞吐量中位数（tokens/s） | 相对无缓存版本 |
| ----- | ----------------:| -------:|
| 无缓存版本 | 305,789          | 1.00 倍  |
| LRU   | 2,048,321        | 6.70 倍  |

两组输出的完整 token ID 序列哈希相同，LRU的收益来自重复片段的结果复用。

## KV Cache

针对自回归生成中存在的历史Token K, V 反复计算问题，引入KVCache，直接在MHA类中加入缓存字段，通过顶层接口传入using_cache决定是否开启缓存，通过顶层向下接口清空缓存。

实验发现加入 KV Cache 后，短前缀的生成速度没有显著提升。因此我把逐步提升前缀长度，固定生成长度，重新做了对照。

测评使用 RTX 4070 Laptop GPU、FP32，同样的参数权重。输入来自 TinyStories 验证集。

| 前缀长度 | 无缓存（tokens/s） | KV Cache（tokens/s） | 倍数   |
| ----:| -------------:| ------------------:| ----:|
| 64   | 135.04        | 146.50             | 1.08 |
| 128  | 153.81        | 145.67             | 0.95 |
| 192  | 137.04        | 137.27             | 1.00 |
| 256  | 138.42        | 152.49             | 1.10 |
| 512  | 108.26        | 141.95             | 1.31 |
| 768  | 56.13         | 137.39             | 2.45 |

当前缀长度增加到 768-token 时才有较明显的收益。小模型中，算子调用、张量构造和采样等开销也会影响结果。

原始计时、配置和源码哈希保存在 [`benchmarks/results/`](./benchmarks/results/)，复测命令见后面的运行说明。

## 跨层残差注意力尝试

在标准 PreNorm Transformer 跑通之后，我参考 [Attention Residuals](https://arxiv.org/abs/2603.15031) 的思路，尝试让模型学习如何组合前面各层的输出。

我的实验版本保留 token embedding 和各个 MHA、FFN 分支的输出作为 `history`。对每个位置，先将这些候选表示做 RMS 归一化，用可学习向量打分，再沿历史维度做 softmax，用得到的权重汇聚原始表示：

$$
\alpha_s = \mathrm{softmax}_s\left(w^\top \mathrm{RMSNorm}(h_s)\right),
\qquad
x = \sum_s \alpha_s h_s
$$

w` 是模型参数，`alpha` 会随输入变化，不能把训练结束后的 `w` 直接当作路由权重。

每个 Transformer Block 有两个长度为 `d_model` 的打分向量，分别用于MHA，FFN 激活值的汇聚。4 层、`d_model=512` 时，只增加了 4,096 个可训练参数。

### 与标准残差的对照

两组实验使用相同的模型尺寸、数据路径、batch size 32 和学习率调度：最大学习率 `1e-3`、最小学习率 `3e-4`、预热 200 步，总训练步数 10,000。

| 第 9901 步的记录 | StdRes  | AttenRes |
| ----------- | -------:| --------:|
| 验证交叉熵       | 1.57275 | 1.56943  |
| 累计耗时（分钟）    | 56.67   | 80.82    |

验证集损失接近，当前实现的 AttenRes 则多用了约 42.6% 的时间。

各层激活 RMS 在 100 个日志点上的平均值如下：

| Transformer 层 | StdRes | AttenRes |
| ------------- | ------:| --------:|
| 0             | 2.288  | 2.094    |
| 1             | 3.908  | 2.319    |
| 2             | 5.909  | 2.185    |
| 3             | 9.942  | 1.613    |

标准残差的激活尺度随层深明显增加，AttenRes 的增长受到显著抑制。但尺度更小不等于时间上的波动更小，也不能据此断言梯度更稳定。

### FLOPs 接近，为什么实际训练更慢

我用 `FlopCounterMode` 统计了一次相同配置的前向和反向：

| 结构       | 一次训练步的已统计 FLOPs |
| -------- | ---------------:|
| StdRes   | 915,230,687,232 |
| AttenRes | 916,337,983,488 |

计数只增加了约 0.12%，但它不覆盖所有逐元素运算、归一化和内存开销。当前 AttenRes 需要反复堆叠历史张量、归一化、打分和加权汇聚；这些操作的开销不能只用新增的矩阵乘法解释。具体瓶颈还需要 profiler 来确认。

打分向量写成 `q @ w.unsqueeze(-1)` 后，投影可以进入矩阵乘法的计数路径，但这仍不是完整的运行成本。统计表中的父模块还包含子模块，不能把两者直接相加。

这次尝试给我的一个实际结论是：新结构确实改变了深层激活的尺度，但在当前小模型、数据和实现下，没有观察到明显的验证损失优势，还付出了额外耗时。后续需要更多随机种子、更深的模型和更细的性能分析。

## 训练记录与测评方式

每次训练会保存配置、运行日志和两份 CSV：

- `config.json`：该次运行实际采用的参数。
- `metrics.csv`：训练步数、累计时间、训练损失、验证损失、学习率。
- `norms.csv`：整体及各层的权重 L2、梯度 L2 和激活 RMS。
- `run.log`：训练状态与检查点保存信息。

# 如何运行

以下命令在仓库根目录、已激活的 Python 环境中执行。本项目使用 Python 3.12，实测环境为 PyTorch 2.11.0 + CUDA 13.0。

```bash
python -m pip install -e .
python -m pip install matplotlib tabulate
```

`matplotlib` 用于绘图，`tabulate` 用于显示 FLOP 统计表。

### 准备数据与分词

将 TinyStories 原始训练文本放到 `data/TinyStoriesV2-GPT4-train.txt`。数据来自 [TinyStories 数据集](https://huggingface.co/datasets/roneneldan/TinyStories)。

```bash
python -m cs336_basics.train_bpe
python -m cs336_basics.run_tokenization
```

这两个入口目前在脚本中指定数据和输出路径，运行前检查对应配置。后续训练使用：

```text
data/result/vocab.txt
data/result/merges.txt
data/train.bin
data/val.bin
```

### 训练

```bash
python -m cs336_basics.run_train \
  --train_path data/train.bin \
  --val_path data/val.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 10000 \
  --max_learning_rate 1e-3 \
  --min_learning_rate 3e-4 \
  --warmup_iters 200 \
  --cosine_cycle_iters 10000 \
  --device cuda \
  --dtype float32 \
  --run_name tinystories
```

残差结构通过 `Transformer(..., using_attenres=False)` 或 `True` 选择，硬编码在代码中。

### 文本生成

```bash
python -m cs336_basics.decoding \
  --bpe_dir data/result \
  --device cuda \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_new_tokens 100
```

启动后输入前缀，输入 `exit` 退出。`decoding.py` 的检查点路径硬编码在代码中。启用缓存时，模型初始化和解码分支的 `using_cache` 都需要开启。

### 课程测试

```bash
python -m pytest
python -m pytest tests/test_model.py tests/test_nn_utils.py
python -m pytest tests/test_optimizer.py tests/test_data.py tests/test_serialization.py
```

课程测试通过 `tests/adapters.py` 调用自己的实现。

### 性能复测

```bash
python benchmarks/benchmark_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --bpe_dir data/result \
  --sample_mb 8 \
  --repeats 5 \
  --output_dir runs/benchmark_bpe

python benchmarks/benchmark_kv_cache.py \
  --checkpoint checkpoints/step_9001.pt \
  --token_file data/val.bin \
  --context_length 1024 \
  --prompt_lengths 256 512 768 \
  --num_prompts 3 \
  --new_tokens 64 \
  --repeats 7 \
  --warmup 3 \
  --modes end_to_end \
  --output_dir runs/benchmark_kv_cache_long
```

更完整的命令、单独模型计算的测评和原始数据说明见 [`benchmarks/results/README.md`](./benchmarks/results/README.md)。

## 参考

- [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) 与仓库中的[作业手册](./cs336_assignment1_basics.pdf)。
- [Attention Residuals](https://arxiv.org/abs/2603.15031)，跨层残差注意力的设计来源。
- [PyTorch FlopCounterMode 源码](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)。
- 原课程代码的许可见 [LICENSE](./LICENSE)。