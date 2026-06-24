# MATH 评估不是换数据集，而是换判分方式——LaTeX-aware 数学等价性判断是如何被逼出来的

## 一、MVP 手把手最小实现

### 1. 第一个愿望：我想在 MATH 验证集上跑 Qwen2.5-Math-1.5B baseline

GSM8K 已经能跑了（`run_benchmarks.py --benchmarks gsm8k`），MATH 也是数学题，直觉上就是改个数据路径的事：

```python
# 天真尝试：把 GSM8K eval 的数据路径指向 MATH
run_gsm8k_eval(
    generator=generator,
    gsm8k_data_path="/root/gpufree-share/data/MATH/validation.jsonl",
    ...
)
```

但一跑就发现问题了。

### 2. 第一个错误：parse_gsm8k_response 从 LaTeX 答案中抽到了错误的数字

MATH 的答案格式和 GSM8K 完全不同：

| 数据集 | 答案示例 | GSM8K parser 抽到的结果 | 期望 |
|--------|---------|------------------------|------|
| GSM8K | `#### 72` | `"72"` | `"72"` |
| MATH | `\dfrac{1}{9}` | `"9"`（最后一个数字） | `1/9` |
| MATH | `\boxed{420}` | `"420"` ✅ 运气好 | `"420"` |
| MATH | `\text{4:30 p.m.}` | `"30"`（最后一个数字） | `4:30 p.m.` |
| MATH | `180\text{ miles}` | `"180"` ✅ 运气好 | `"180"` |
| MATH | `\frac{1}{4}` | `"4"`（最后一个数字） | `"1/4"` |

`parse_gsm8k_response` 的核心逻辑是 **"从文本中取最后一个数字"**——这对 GSM8K 的 `#### 72` 格式恰好有效，但对 MATH 的 LaTeX 表达式完全是灾难。

```text
parse_last_number("\dfrac{1}{9}") → "9"
numbers_equal("9", "1/9") → False  ← 数学上相等，但 parser 判错
```

**核心矛盾**：GSM8K 的 parser 假设"答案就是文本里的最后一个数字"，MATH 的答案却是 LaTeX 数学表达式，需要符号级等价性判断。

### 3. 深入问题：LaTeX 答案不止一种合法写法

同一个数学答案可以有多种 LaTeX 表示：

| 数学值 | LaTeX 写法 1 | LaTeX 写法 2 | 纯文本 |
|--------|-------------|-------------|--------|
| 1/9 | `\frac{1}{9}` | `\dfrac{1}{9}` | `1/9` |
| 1/4 | `\frac{1}{4}` | `\dfrac{1}{4}` | `1/4` |
| 3 | `3` | `\frac{6}{2}` | `3` |
| √2 | `\sqrt{2}` | `2^{\frac{1}{2}}` | — |

如果只做字面字符串比较：
- `\frac{1}{9}` ≠ `\dfrac{1}{9}`（字面不同，但数学相等）
- `\frac{1}{9}` ≠ `1/9`（字面不同，但数学相等）
- `\boxed{420}` ≠ `420`（字面不同，但数学相等）

**需要 LaTeX 解析 + 符号数学等价性判断**。

### 4. 已有答案：`reasoning/rewards.py` 的 grader

往项目里一看，`cs336_alignment/reasoning/rewards.py` 其实已经实现了完整的 MATH 判分体系：

```text
grade(model_answer, ground_truth)
  ├── grade_answer_mathd()    → 字符串归一化后比较
  └── grade_answer_sympy()    → sympy 符号等价性判断
      ├── _normalize()        → 去空格、去 boxed、去冗余 LaTeX
      ├── split_tuple()       → 处理元组答案 (a, b)
      ├── are_equal_under_sympy() → sympy.simplify 判断等价
      └── is_latex_equal()    → 兜底用 math_verify 库做 LaTeX 语义比较
```

它的判分能力：

| 对比场景 | `grade_answer_mathd` | `grade_answer_sympy` | `is_latex_equal` |
|---------|---------------------|---------------------|-----------------|
| 纯数字 42 vs 42 | ✅ 字符串相等 | ✅ | ✅ |
| `\frac{1}{9}` vs `1/9` | ❌ 字面不等 | ✅ sympy 化简 | ✅ |
| `\frac{1}{9}` vs `\dfrac{1}{9}` | ❌ 字面不等 | ✅ sympy 化简 | ✅ |
| `\sqrt{2}` vs `2^{0.5}` | ❌ 字面不等 | ✅ sympy | ✅ |
| `\text{4:30 p.m.}` vs `4:30 p.m.` | ✅ 字符串相等 | ✅ | ✅ |

这不是专门为评估写的，而是 GRPO 训练时用的 reward function。它被"训练时需要判模型答对还是答错"这个需求逼出来的。

### 5. 还需要什么

有了 grader，还需要：

1. **Data loader** — 读取 MATH 的 jsonl 格式（`problem`, `level`, `subject`, `unique_id`, `answer`）
2. **Prompt builder** — 用 `r1_zero.prompt` 模板包装 problem
3. **Answer extractor** — 从模型输出中抽 `<answer>...</answer>` 标签里的内容
4. **Eval orchestrator** — 串联生成 → 提取 → 判分 → 汇总

这些和 GSM8K eval 的骨架完全一样，只是数据格式和判分函数不同。

## 二、八股概念基础知识点

### LaTeX 与符号数学等价性

**为什么 `\frac{1}{9}` 和 `1/9` 需要特殊处理？**

从字符串角度看，它们完全不同。但在数学上，它们表示同一个有理数。

解决思路：

```
模型输出 "\frac{1}{9}"
  ↓
归一化：去掉空格、去掉 \boxed 包装、统一分数格式
  ↓
sympy.simplify(1/9) - sympy.simplify(1/9) = 0  → 等价
```

`grade_answer_sympy` 的实现策略：
1. **归一化**：`_normalize()` 去掉多余空格、统一 `\dfrac` → `\frac`、去掉 `\boxed` 外壳
2. **字符串快捷比较**：如果归一化后完全相同，直接返回 True（避免 sympy 开销）
3. **分类型比较**：
   - 分数 vs 分数：要求严格相等（不化简，`1/3` ≠ `2/6`）
   - 整数 vs 非整数：要求严格相等（不允许 sympy 化简）
   - 其他：用 `sympy.simplify(a - b) == 0` 判断等价
4. **兜底**：`is_latex_equal` 调用 `math_verify` 库做 LaTeX 语义解析

### MATH 数据集结构

原版 MATH 数据集（Hendrycks et al., 2021）：
- 12,500 道数学竞赛题（train 7,500 / test 5,000）
- 7 个学科：Prealgebra, Algebra, Number Theory, Geometry, Counting & Probability, Precalculus, Intermediate Algebra
- 5 个难度等级：Level 1（最简单）到 Level 5（最难）
- 答案用 LaTeX 或纯文本表示

本项目使用的版本：
- `train.jsonl` — 7,500 条（来自原版 train）
- `validation.jsonl` — 5,000 条（来自原版 test）
- `sft.jsonl` — 1,767 条（已用 r1_zero 模板格式化的 SFT 数据）

### TextGenerator Protocol 的复用价值

GSM8K eval 和 MATH eval 共享完全相同的推理后端需求：

```
GSM8K eval                     MATH eval
    │                              │
    └──────────┬───────────────────┘
               │
        TextGenerator Protocol
         (generator.generate)
               │
        ┌──────┴──────┐
     HF backend    vLLM backend
```

所以不需要为 MATH 重复实现生成后端。`eval/math.py` 只依赖 `generator: TextGenerator`，和 `eval/gsm8k.py` 一样。

### Interview phrasing

> "MATH 评估和 GSM8K 评估的表面区别是数据集不同，但核心区别是答案判分方式。GSM8K 的答案永远是最后一个数字（`#### 42`），但 MATH 的答案用 LaTeX 表示（`\frac{1}{9}`、`\boxed{420}`、`\text{4:30 p.m.}`），需要符号级等价性判断。判分函数 `grade_answer_sympy` 先用字符串归一化做快速比较，再 fallback 到 sympy 化简做数学等价性判断。整个 eval 管线复用 `TextGenerator` Protocol，不需要重新实现生成后端。"

## 三、排障过程实践

### 现象 1：MATH data loader 字段和 GSM8K 不同

GSM8K 的字段是 `question` / `answer`，MATH 的字段是 `problem` / `level` / `subject` / `unique_id` / `answer`。

```python
# GSM8K
row["question"]  # "Natalia sold clips to..."
row["answer"]    # "Natalia sold 48/2 = <<48/2=24>>24... #### 72"

# MATH
row["problem"]   # "What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$..."
row["answer"]    # "420"
row["level"]     # "Level 3"
```

**修复**：`load_math_examples` 用 `row["problem"]` 而不是 `row["question"]`。

### 现象 2：r1_zero 模板的 prompt 格式和 GSM8K 不同

GSM8K eval 用的是一个简单的零样本 prompt：
```
Solve the following grade school math problem. ...
```

但 MATH 评估应该和训练时保持一致，用 `r1_zero.prompt` 模板：
```
A conversation between User and Assistant...
User: {problem}
Assistant: <think>
```

这样模型会生成 `<think> reasoning </think> <answer> answer </answer>` 格式的输出。

**修复**：MATH eval 的 `make_math_prompt` 用 `r1_zero.prompt` 模板格式化。

### 现象 3：`grade()` 的 `fast=False` 模式可能很慢

`is_latex_equal` 调用 `math_verify` 库，内部做了 LaTeX 语义解析 + 符号比较，超时 1 秒。

对于 5000 条全量验证集，如果每条都走 `is_latex_equal` 可能会慢很多。默认 `fast=True`（只用 `grade_answer_mathd` + `grade_answer_sympy`），这两个都是字符串归一化 + sympy 化简，通常 < 10ms 每条。

### 现象 4：模型输出中可能没有 `<answer>` 标签

如果模型没按 r1_zero 格式输出（例如只是纯文本），`extract_math_answer` 会返回 None，grade 会判 False。

这是预期的——没有格式的答案等同于答错。在最终 summary 中我会输出 `format_rate` 来跟踪格式遵守率。

### 完整可运行命令

```bash
# vLLM 后端（推荐）
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine vllm \
    --benchmarks math \
    --math_path /root/gpufree-share/data/MATH/validation.jsonl \
    --output_dir outputs/baseline_math \
    --max_new_tokens 1024 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 4096

# HF 后端（fallback）
uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --engine hf \
    --benchmarks math \
    --math_path /root/gpufree-share/data/MATH/validation.jsonl \
    --output_dir outputs/baseline_math_hf \
    --device cuda:0 \
    --hf_batch_size 8 \
    --max_new_tokens 1024
```
