# ai新增
# 简历性能数据的复测

两份脚本已放到 WSL 项目的 `/home/aixkai/336/benchmarks/` 中。进入项目根目录，并激活已有的 336 Python 环境后运行。

## BPE 缓存

```bash
cd ~/336
python benchmarks/benchmark_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --bpe_dir data/result \
  --sample_mb 8 \
  --repeats 5 \
  --warmup 1 \
  --output_dir runs/benchmark_bpe
```

脚本直接读取当前 `Tokenizer.encode`，并在内存中恢复你保存在同一个文件注释里的缓存版 `encode`，不会改写该文件。四组比较如下：

- `current`：当前生效的无缓存实现。
- `optimized_no_cache`：使用缓存版函数，但把缓存容量设为 0。
- `optimized_lru_cold`：每次测速前清空缓存，处理该次全部文档时持续复用。
- `optimized_lru_warm`：先编码同一份样本，再计时；用于观察预热缓存收益，单独报告。

约 8 MiB 样本截止于完整的 `<|endoftext|>` 文档边界。采用与 `run_tokenization.py` 一致的逐文档调用方式，验证各版本的全部 token ID 序列哈希相同。计时只包含单进程编码，不包含文件读取、初始化、文件写入或多进程启动。

## KV Cache

```bash
cd ~/336
python benchmarks/benchmark_kv_cache.py \
  --checkpoint checkpoints/step_9001.pt \
  --token_file data/val.bin \
  --prompt_lengths 64 128 192 \
  --num_prompts 3 \
  --new_tokens 64 \
  --repeats 7 \
  --warmup 3 \
  --modes model_only end_to_end \
  --output_dir runs/benchmark_kv_cache
```

脚本读取你的 `Transformer` 和模型检查点，按检查点识别 StdRes / AttenRes 并严格加载权重。每轮先清空缓存，再统一设置所有注意力层的缓存开关。

- `model_only`：使用同一段 TinyStories token 流，按已有正确 token 逐步输入，比较模型计算耗时；输入已在 GPU 上，不含随机采样。
- `end_to_end`：使用真实前缀，沿用 `decoding.py` 的温度与 top-p 采样步骤；包含前缀处理、输入张量创建、设备传输和采样，不包含原始文本分词及最终文本拼接。

每组强制生成 64 个 token，不因 EOS 提前结束。总长度不超过模型的 256-token 上下文，两个版本都使用全部历史，不启用滑动窗口。CUDA 计时前后同步，缓存开关的测试顺序交替，每组预热 3 次并测量 7 次。

补测较长前缀：

```bash
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

这组只在模型初始化时将 RoPE 预计算表扩展为 1024 个位置，模型权重相同，不改写源码。检查点在 256-token 上下文上训练，因此该补测用于比较计算性能，不能据此判断长上下文生成质量。更长正确性检查还比较了 512-token 前缀后新增 token 的 logits。

脚本先验证完整前向与缓存前向的 logits 接近，检查每层缓存长度；另记录每轮有无缓存生成的 token ID 是否相同。

## 绘图

直接用本次已测出的记录重新画图：

```bash
cd ~/336
python benchmarks/plot_benchmark_results.py \
  --bpe_dir benchmarks/results/bpe \
  --kv_dir benchmarks/results/kv_cache benchmarks/results/kv_cache_long \
  --output_dir benchmarks/results/figures
```

如果按上面的测评命令重新运行并把日志存到 runs，则使用：

```bash
python benchmarks/plot_benchmark_results.py \
  --bpe_dir runs/benchmark_bpe \
  --kv_dir runs/benchmark_kv_cache runs/benchmark_kv_cache_long \
  --output_dir runs/benchmark_figures
```

生成三张 PNG 与包含全部图的 benchmark_results.pdf。BPE 主图使用每轮空缓存的版本；吞吐量图显示中位数和重复测量的四分位区间。KV Cache 加速倍数使用“无缓存中位数耗时 / 有缓存中位数耗时”。

## 输出文件

每个目录包含：

- `benchmark_trials.csv`：每轮原始计时及吞吐量。
- `benchmark_summary.csv`：中位数、最快/最慢耗时与加速倍数。
- `benchmark_config.json`：输入、设备、配置、源码哈希和正确性检查结果。

BPE 目录额外保存 `measured_cached_encode.py`，这是从项目注释中提取并实际执行的函数，方便核对。

本次已运行的结果保存在同级的 `bpe/`、`kv_cache/` 和 `kv_cache_long/` 目录。简历只采用明确说明条件的实测值，预热缓存结果不作为通用加速倍数。
