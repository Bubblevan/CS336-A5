# GSM8K 评估从逐条生成到批量推理：Batched HF / vLLM 是如何被逼出来的

## 一、MVP 手把手最小实现

### 1. 第一个愿望：我希望 baseline 模型能在 GSM8K 上跑出得分

```python
def run_gsm8k_eval(
    model, tokenizer, gsm8k_path, *, device, max_new_tokens=512,
):
    examples = load_gsm8k_examples(gsm8k_path)
    for example in examples:
        prompt = make_gsm8k_prompt(example["question"])
        model_output = generate_one(model, tokenizer, prompt, device=device, ...)
        pred = parse_gsm8k_response(model_output)
        gold = parse_gsm8k_gold_answer(example["answer"])
        is_correct = numbers_equal(pred, gold)
        ...
```

先实现了最直觉的事情：**一条 prompt → 一次 `model.generate()` → 拿到输出 → 下一条**。

### 2. 跑起来以后 GPU 利用率极低：decode 阶段大部分 time budget 闲置

跑 1319 条 GSM8K 时 `nvidia-smi` 看到 GPU 利用率只有 15-25%。

原因：

```text
每一条的时序：
  [prefill] → [decode token 1] → [decode token 2] → ... → [decode token N] → [下一题]
    利用率 100%     利用率 ~10%      利用率 ~10%                    清空 KV cache

1319 题 = 1319 次 prefill + 1319 × 512 次单 token decode
decode 阶段 GPU 算力闲置，瓶颈在 memory bandwidth，芯片大部分单元空闲
```

**核心矛盾**：我需要等 1319 次迭代全部串行跑完才能看总分。每条 1-2 秒，总计半小时以上。如果想调 prompt 或采样参数，再等半小时。

### 3. 第一次改进：HuggingFace batched generation（left padding + batch decode）

把 `make_gsm8k_prompt` 构造的 prompt 攒成一个 list，一次 `model.generate()` 处理多道题：

```python
encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
generated_ids = model.generate(**encoded, max_new_tokens=512, ...)
new_token_ids = generated_ids[:, input_length:]
batch_outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
```

但这有两个陷阱：

**陷阱 1：decoder-only 模型必须 left padding**

```text
prompt A: [tok1, tok2, tok3, <pad>, <pad>]   ← 左边 padding
prompt B: [tok1, tok2, tok3, tok4, tok5  ]

如果 right padding：
prompt A: [tok1, tok2, tok3, <pad>, <pad>]
prompt B: [tok1, tok2, tok3, tok4, tok5  ]
                                ^—— 右边 padding 的位置不同，position id 错位
```

`tokenizer.padding_side = "left"` 是 decoder-only generation 的标准配置。

**陷阱 2：stop string 需要手动截断**

HF `generate()` 没有 `stop` 参数。所以要加一步：

```python
for text in batch_outputs:
    for stop in stop_strings:
        if stop in text:
            text = text[:text.find(stop)]
```

封装在 `truncate_at_stop_strings()` 函数里。

### 4. 再次遇到瓶颈：HF batch 仍然不够快——vLLM offline batched inference

HF batched 模式已经大幅提速（batch_size=8 时约 4-5x），但 GPU 仍然有空闲。vLLM 用 **paged attention + continuous batching** 让 GPU 全程满载：

```python
class VLLMGenerationConfig:
    def __post_init__(self):
        import vllm
        self.llm = vllm.LLM(model=self.model_id_or_dir, ...)
    
    def generate(self, prompts, *, max_new_tokens, stop_strings=None):
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, stop=stop_strings,
        )
        request_outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        return [output.outputs[0].text for output in request_outputs]
```

区别：

| | HF batched generate | vLLM offline inference |
|---|---|---|
| Batch 内调度 | 整批同步 decode | 每个 sequence 独立调度 |
| KV cache | 每批次重新分配 | Paged attention，按 page 分配 |
| Decode 等待 | 慢的等快的，快的等慢的 | 快的继续 decode，不等待 |
| GPU 利用率 | 60-70% | 95%+ |

### 5. TextGenerator Protocol 出现：evaluation 层不关心后端

一开始 `run_gsm8k_eval` 的参数是 `(model, tokenizer, device, ...)`。切换到 vLLM 时发现函数签名完全不同：

```python
# HF 版本
run_gsm8k_eval(model=model, tokenizer=tokenizer, device="cuda:0", ...)

# vLLM 版本
# model/tokenizer/device 都不需要，需要 model_id_or_dir
run_gsm8k_eval(llm=llm_instance, ...)
```

两个分支在调用侧产生了 if/else 扩散。我需要一个统一接口：

```python
class TextGenerator(Protocol):
    def generate(
        self, prompts: list[str], *, max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[str]: ...
```

然后 `run_gsm8k_eval` 只依赖 `generator: TextGenerator`，不再知道 HF 还是 vLLM：

```python
def run_gsm8k_eval(generator: TextGenerator, gsm8k_data_path, *, ...):
    prompts = [make_gsm8k_prompt(ex["question"]) for ex in examples]
    model_outputs = generator.generate(prompts, max_new_tokens=512)
    # 统一返回 list[str]，和 backend 无关
```

这不是为了设计模式，而是为了让 `eval/gsm8k.py` **不需要改动**就能切换后端。后续如果换 TensorRT-LLM / SGLang，也只有 `build_generator()` 一处需要改。

### 6. 最终效果：Qwen2.5-Math-1.5B on GSM8K

全部 1319 条 GSM8K test 集跑完后的 summary：

```
GSM8K summary:
  benchmark: gsm8k
  split: test
  data_path: data/gsm8k/main/test-00000-of-00001.parquet
  num_examples: 1319
  correct: 933
  accuracy: 0.7074 (70.7%)
  parse_rate: 1.0
  gold_parse_rate: 1.0
  max_new_tokens: 512
  elapsed_seconds: 22.83
  examples_per_second: 57.77

Saved summary to: outputs/baseline_gsm8k/summary.json
```

关键指标解读：

| 指标 | 值 | 说明 |
|---|---|---|
| **accuracy** | 70.7% | Qwen2.5-Math-1.5B 零样本直接答小学数学题的正确率。说明这个基座模型本身已经有不错的数学推理能力 |
| **parse_rate / gold_parse_rate** | 1.0 | 所有 1319 条的模型输出和 ground truth 都成功提取到了数字，没有格式失败。`<answer>...</answer>` 标签和 `####` 两种格式的 parser 都覆盖到了 |
| **elapsed_seconds** | 22.83s | vLLM offline batched 只用了 23 秒就跑完了 1319 题。对比逐条 generate 的预估时间：1319 × 1.5s ≈ 33 分钟，提速约 **86 倍** |
| **examples_per_second** | 57.77 | 每秒钟处理 58 道题，包含 prefill + decode 全过程 |

提速的核心不是模型变快，而是**让 GPU 在 decode 阶段不再空闲**——vLLM 的 continuous batching + paged attention 把大量等待时间重叠掉了。

---

## 二、八股概念基础知识点

### Left padding vs Right padding

| | Left padding | Right padding |
|---|---|---|
| Encoder 模型 | ❌ 影响 attention | ✅ 标准做法 |
| Decoder-only 生成 | ✅ 保证 position id 连续 | ❌ 生成 token 位置错乱 |

**判断标准**：如果模型是 `CausalLM`（单向 attention），生成时用 left padding。如果模型是 `EncoderModel`（双向 attention），用 right padding。

### Paged Attention（vLLM 的核心）

传统 attention：KV cache 连续分配，prefill 时就 max_seq_len 预留空间 → 显存碎片 + 浪费。

Paged attention：按固定大小的 page（block）分配 KV cache——类似操作系统的虚拟内存分页。好处：
- 按需分配，不浪费 unused token 位置的显存
- 物理 page 不连续，消除碎片
- batch 内不同 sequence 共享 prefill 阶段的 KV cache（prefix caching）

### Continuous batching

传统 batching：等整批全部 decode 完 → 换下一批 → prefill。

Continuous batching：batch 内一条 sequence 生成到 eos 后立即移除，空位由新请求的 prefill 补上。新请求不需要等到整批结束才进来。

所以 vLLM 的 batch 大小在运行中是**动态的**：prefill 阶段新 sequence 加进来，decode 快的 sequence 继续，eos 的退出。

### Interview phrasing

> "GSM8K 评估的瓶颈在推理效率，不是模型能力——单条串行 generate 下 GPU decode 阶段的利用率很低。第一步用 HF batched + left padding 把利用率提到 60%；第二步切 vLLM offline inference，paged attention + continuous batching 让利用率到 95%+。后端的切换由一个 `TextGenerator` Protocol 抽象，`run_gsm8k_eval` 不依赖具体引擎——下次换引擎（TensorRT-LLM / SGLang）也只改 `build_generator` 一处。"

---

## 三、排障过程实践

### 现象 1：逐条 generate 时 GPU 利用率低

```text
nvidia-smi 显示 GPU-Util 15-25%，但显存占满
```

**排查**：
1. 确认 `max_new_tokens` 设了足够大（512），不是生成太短
2. 打印每条 `model.generate()` 耗时：prefill 快（~2ms/token），decode 慢（~20ms/token）
3. core 是 **memory bound**：decode 阶段每个 step 只读一个 token 的 KV cache，计算量极低，卡在 HBM 带宽

**结论**：不是模型大导致慢，是串行导致 GPU 大部分时间在空转等待 memory fetch。

### 现象 2：切换到 HF batched 后第一次跑结果变差

**原因**：没有设置 `padding_side = "left"`。right padding 下 position id 为 padding 位置分配了错误的位置嵌入。

```python
# 修复
tokenizer.padding_side = "left"
```

### 现象 3：vLLM 报 `max_model_len` 参数错误

```
TypeError: VLLMGenerationConfig got unexpected keyword argument 'max_model_len'
```

**原因**：`run_benchmarks.py` 传给 `VLLMGenerationConfig` 的参数名 `max_model_len` 和 dataclass 字段名 `max_model_length` 不一致。argparse 的参数名和 dataclass 参数名之间少了一次映射。
**修复方法**：统一用 `max_model_length` 作为 dataclass 字段名，argparse 侧 `--max_model_len` 传进来时在 `build_generator()` 中做一次重命名。

### 现象 4：`flash-attn` 无法安装导致模型加载失败

```text
ImportError: FlashAttention2 has been toggled on, but it cannot be used
```

**排查**：`get_model_and_tokenizer()` 在 CUDA 设备上硬编码 `attn_implementation="flash_attention_2"`，但 `uv sync --no-install-package flash-attn` 排除了 flash-attn。

**修复**：引入 `_get_attn_implementation(device)` 函数，import try 检测 `flash_attn` 是否可导入，不可用时 fallback 到 `"eager"`。不影响 eval 正确性，只是 decode 速度稍慢。

### 现象 5：`gsm8k_data_path` vs `gsm8k_path` 参数名不匹配

```
TypeError: run_gsm8k_eval() got an unexpected keyword argument 'gsm8k_data_path'
```

**原因**：V0 的函数参数是 `gsm8k_path`，V1 修改为 `gsm8k_data_path`，但 `run_benchmarks.py` 的调用侧没有同步更新。

**教训**：修改函数签名时 grep 所有调用方。Protocol 的出现目的之一就是让这类接口变更集中在 `build_generator()` 里。

### 现象 6：`RuntimeError: Cannot re-initialize CUDA in forked subprocess`

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

**背景链路**：

```text
uv run python → 提前初始化 CUDA
  → 创建 vLLM EngineCore 子进程（fork）
    → 子进程调用 torch.cuda.set_device()
      → RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**根因**：

Linux 默认多进程创建方式 `fork()`：
- fork 会复制父进程全部内存、CUDA 上下文到子进程
- CUDA 不允许子进程复用/重新初始化已存在的 CUDA 上下文，直接抛错

vLLM v0.22.x 默认开启 V1 异步多进程引擎：
- EngineCore 单独开子进程调度
- 父进程提前加载过 CUDA，fork 子进程后初始化 GPU 直接失败

**修复**：设置环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，让 vLLM 用 `spawn` 创建子进程（不复制父进程 CUDA 上下文）：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine vllm \
    --gsm8k_path data/gsm8k/main/test-00000-of-00001.parquet \
    --output_dir outputs/baseline_qwen_math_vllm \
    --max_new_tokens 512 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 2048
```

| | fork | spawn |
|---|---|---|
| 子进程启动方式 | 复制父进程全部内存 | 新 Python 解释器进程 |
| CUDA 上下文 | 复制到子进程，不可重初始化 | 全新，可以初始化 |
| 启动速度 | 快 | 慢（重新 import） |
| 兼容性 | CUDA 不兼容 | 兼容 |

### 现象 7：vLLM V1 Torch Inductor / Triton 编译失败

```text
Triton 底层 CUDA 工具库编译 → 返回非零退出码 → Inductor 抛出异常 → 引擎初始化终止
```

**背景**：解决 spawn 问题后，vLLM V1 引擎初始化 → profile_run 预热模型 → 默认开启 Torch Inductor 编译优化 → Triton 初始化 NVIDIA 驱动 → 编译 `cuda_utils.c` 辅助库。

**根因**：容器环境缺少编译依赖：
- Python 头文件缺失（`Python.h`）
- CUDA 开发库缺失（`cuda_runtime.h` 等）

**修复**：

```bash
# 安装基础编译工具 + Python 开发头文件
apt-get update && apt-get install -y build-essential python3-dev

# 安装 CUDA 开发运行时库
apt-get install -y cuda-cudart-dev-12-0
```

安装后重新运行，vLLM V1 引擎即可完成 Inductor 编译、初始化成功。

### 完整可运行命令

```bash
# 安装编译依赖（仅首次需要）
apt-get update && apt-get install -y build-essential python3-dev cuda-cudart-dev-12-0

# 跑 vLLM baseline 评估
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine vllm \
    --benchmarks gsm8k \
    --gsm8k_path data/gsm8k/main/test-00000-of-00001.parquet \
    --output_dir outputs/baseline_gsm8k \
    --max_new_tokens 512 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 2048
```

输出结果：

```
GSM8K summary:
  benchmark: gsm8k
  split: test
  data_path: data/gsm8k/main/test-00000-of-00001.parquet
  num_examples: 1319
  correct: 933
  accuracy: 0.7074
  parse_rate: 1.0
  gold_parse_rate: 1.0
  max_new_tokens: 512
  elapsed_seconds: 22.83
  examples_per_second: 57.77

Saved summary to: outputs/baseline_gsm8k/summary.json
```
