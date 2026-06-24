# SFT 数据预处理与基线评估

## 一、数据准备

### 1. 原始数据来源

两个 SFT 训练数据集的原始文件存放在共享目录 `/root/gpufree-share/data/`：

| 数据集 | 原始路径 | 格式 | 记录数 |
|--------|----------|------|--------|
| GSM8K (main) | `gpufree-share/data/gsm8k/main/` | Parquet | train 7,473 / test 1,319 |
| TULU-3 SFT Personas Math | `gpufree-share/data/tulu-3-sft-personas-math/data/` | Parquet | train 149,960 |

### 2. 为什么不在 `cs336/data/` 下直接放处理后的文件

一开始我在 `scripts/prepare_sft_data.py` 里把 `DATA_DST` 设成了 `cs336/data/`。

跑完后发现 TULU 的 JSON 有 **788 MB**——这个体积 git 无法追踪、`cs336/data/` 会膨胀到难以管理。

所以决策改为：
1. 处理后的数据统一放到 `/root/gpufree-share/data/sft/`（共享存储，不占项目空间）
2. `cs336/data/` 下用软链接引用过去

### 3. 软链接布局

```
# 真实数据（共享存储，gitignored）
gpufree-share/data/sft/
├── gsm8k/
│   ├── train.json      7,473 条
│   ├── train.jsonl
│   ├── test.json       1,319 条
│   └── test.jsonl
└── tulu-3-sft-personas-math/
    ├── train.json    149,960 条
    └── train.jsonl

# 项目内软链接（路径不变，训练代码不用改）
cs336/data/gsm8k/train.json          → ../../../../gpufree-share/data/sft/gsm8k/train.json
cs336/data/gsm8k/train.jsonl
cs336/data/gsm8k/test.json
cs336/data/gsm8k/test.jsonl
cs336/data/tulu-3-sft-personas-math  → ../../../gpufree-share/data/sft/tulu-3-sft-personas-math
```

注意：GSM8K 软链接在 `data/gsm8k/` 子目录内，需要 4 层 `..`；TULU 直接在 `data/` 下，需要 3 层。

### 4. SFT 数据格式

每条记录统一为 JSON / JSONL：

```json
{
  "question": "数学题原文",
  "answer":   "GSM8K 原始答案（含 #### 42）或空（TULU）",
  "prompt":   "喂给模型的 prompt（GSM8K 已用 r1_zero 模板格式化）",
  "response": "期望的 SFT 回答"
}
```

#### GSM8K 字段说明

原始 GSM8K 的 answer 格式：
```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
#### 72
```

转换步骤：
1. **去掉 `<<expr=result>>` 标记**（正则 `r"<<[^>]+>>"` → 空字符串）
   - 有坑：如果替换为 `result` 会得到 `2424`（`<<48/2=24>>24` → `2424`）
   - 正确做法：直接删除标记，保留后面的结果文字 → `24`
2. **提取 `####` 后面的数字作为最终答案**
3. **包装成 `<think> reasoning </think> <answer> answer </answer>` 格式**
4. **prompt 用 `r1_zero.prompt` 模板格式化**

#### TULU-3 字段说明

原始 TULU 的 `messages` 字段是 list of dict：
```json
[
  {"role": "user", "content": "数学题..."},
  {"role": "assistant", "content": "详细解答..."}
]
```

转换：直接把 user content 作为 `prompt`，assistant content 作为 `response`。

### 5. 运行方式

```bash
uv run python scripts/prepare_sft_data.py
```

脚本逻辑：
1. 读取 `gpufree-share/data/gsm8k/main/*.parquet` → 清洗 → 写入 `gpufree-share/data/sft/gsm8k/`
2. 读取 `gpufree-share/data/tulu-3-sft-personas-math/data/*.parquet` → 拆 messages → 写入 `gpufree-share/data/sft/tulu-3-sft-personas-math/`

---

## 二、TULU 基线评估

### 1. 动机

在 SFT 训练之前，先看看 base model（Qwen2.5-Math-1.5B）在 TULU 数据上的表现。这样才能知道 SFT 之后提升了多少。

### 2. 评估脚本

**`scripts/evaluate_tulu_baseline.py`**

#### 评估流程

```
加载 TULU 数据
  ↓
从 ground truth response 中提取最终答案
  （优先 "Final Answer:" 模式 → \boxed{} → 最后一个数字）
  ↓
用 base model 生成对每个 prompt 的响应
  （支持 HF 或 vLLM 后端）
  ↓
从模型输出中提取答案
  ↓
与 ground truth 比对
  （先 parse_last_number 做数值比较，兜底字符串相等）
  ↓
输出 accuracy + parse rate + 耗时
```

#### 答案提取策略

TULU 的 assistant response 末尾有稳定的格式：
```
Final Answer: 42. I hope it is correct.
```

提取正则：`r'Final Answer:\s*(.+?)(?:\.\s*I hope it is correct\.|\.\s*$|$)'`

但如果模型输出不是 TULU 风格（比如只有数字或 LaTeX 表达式），兜底策略：
- `\boxed{42}` → 提取 `42`
- 纯文本 → 取最后一个数字

#### 答案比对

不能用 `reasoning/rewards.py`（依赖未安装的 `latex2sympy2_extended`），改用 `eval/parsers.py`：

```python
pred_num = parse_last_number(pred)    # "The answer is 42" → "42"
gold_num = parse_last_number(gold)    # "42" → "42"
numbers_equal(pred_num, gold_num)     # True
```

### 3. 使用方式

```bash
# HF 后端（200 条样本）
uv run python scripts/evaluate_tulu_baseline.py \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --limit 200 --device cuda:0 --max_new_tokens 1024

# vLLM 后端（更快）
uv run python scripts/evaluate_tulu_baseline.py \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine vllm --vllm_device cuda:0 --limit 200
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | `/root/gpufree-share/models/Qwen2.5-Math-1.5B` | 模型路径 |
| `--data_path` | `data/tulu-3-sft-personas-math/train.json` | TULU 数据 |
| `--limit` | `None`（默认全量） | 随机采样的评估样本数 |
| `--engine` | `hf` | `hf` 或 `vllm` |
| `--max_new_tokens` | 1024 | 生成最大 token 数 |
| `--output_path` | `outputs/tulu_baseline.json` | 结果输出路径 |

#### TULU-3 基线结果（Qwen2.5-Math-1.5B）

```text
Samples:      15000
Accuracy:     0.2199  (3298/15000)
Parse rate:   1.000
Time:         1146.7s  (13.1 examples/s — vLLM)
```

模型在 15,000 条 TULU 数学题上准确率 21.99%。这可以作为 SFT 训练前的 baseline，之后用来衡量 SFT 带来的提升。

---

## 三、RLVR-MATH 蒸馏数据

### 1. 数据集说明

- **来源**: RLVR-MATH（基于 MATH benchmark）
- **数量**: 7,500 条
- **格式**: 每条包含 4-shot 的 few-shot prompt + 一个 target 数学题
- **ground_truth**: 干净的标准答案（数字或表达式）
- **用途**: 通过 DeepSeek API 蒸馏出 reasoning traces，作为 SFT 训练数据

### 2. 蒸馏流程

```bash
# 试跑 50 条
uv run python scripts/distill_rlvr_math.py --max_samples 50 --max_workers 10

# 全量蒸馏（支持 --resume 断点续传）
uv run python scripts/distill_rlvr_math.py --max_workers 100
```

蒸馏脚本自动：
1. 从 `messages` 中提取最后一个 target 问题
2. 用 r1_zero 模板包装，调用 DeepSeek API
3. 校验 API 的回答是否匹配 `ground_truth`（答错的跳过）
4. 保存为 SFT 格式（`prompt` + `response`）
5. 创建 symlink：`cs336/data/rlvr-math-distilled → gpufree-share/data/sft/rlvr-math-distilled`

### 3. 蒸馏质量

试跑 50 条结果：

```text
Total processed:  50
Correct (saved):  29  (58%)
Skipped (wrong):  21  (42%)
Accuracy rate:    58.0%
```

- 保留的数据都有完整的 `<think>` 推理链 + `<answer>` 答案格式
- 42% 的跳过率说明校验机制有效——答错的数据不会被小模型学

### 4. RLVR-MATH 基线结果（Qwen2.5-Math-1.5B）

**不需要等蒸馏完成**，基线评估直接使用原始数据中的 `ground_truth`。

```bash
uv run python scripts/evaluate_rlvr_baseline.py \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine vllm --vllm_device cuda:0 --limit 200
```

等待补充基线结果。

### 4. 避坑记录

1. **`reasoning/rewards.py` 的 import 链** — 该文件顶部 `from latex2sympy2_extended import latex2sympy`，该包未在项目中安装，所以不能直接 import。改用 `eval/parsers.py` 的轻量版数字提取和比对。
2. **`<<...>>` 清洗** — GSM8K 的 `<<48/2=24>>24` 如果替换为 `\1`（即 `24`）会得到 `2424`。应直接删除 `<<...>>` 标记。
3. **软链接路径深度** — `cs336/data/gsm8k/` 内的 symlink 比 `cs336/data/` 根目录深一层，`..` 数量不同。
