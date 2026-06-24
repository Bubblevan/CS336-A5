# SFT 训练基建：从数据到 Loss 的决策链

## 开篇：SFT 到底在做什么？

**监督微调（Supervised Fine-Tuning, SFT）** 的任务是：
> 给模型一批 `(问题, 标准答案)` 对，让模型学会输出**类似标准答案**的回答。

数学上，就是在标准答案的每个 token 位置上最大化模型预测概率——也就是**最小化负对数似然（Negative Log-Likelihood, NLL）**。

本文从**一个具体的愿望出发**——"让 Qwen2.5-Math-1.5B 在 MATH 上从 22.5% 提升到更高"——追踪每一行代码、每一个 bug 是怎么被逼出来的。

---

## 一、MVP 手把手最小实现

### 1. 第一个愿望：我想让模型学会回答数学题

我有一批训练数据（MATH 蒸馏数据），每条包含：

```json
{
  "question": "数学题原文",
  "answer": "标准答案",
  "prompt": "r1_zero 模板格式化后的问题",
  "response": "<think> 推理过程 </think> <answer> 答案 </answer>"
}
```

我最朴素的想法：

```python
# ❌ 伪代码——这行不通，但先写出来看看为什么
for each (prompt, response) in dataset:
    full_text = prompt + response                     # 拼接
    input_ids = tokenizer.encode(full_text)            # 分词
    logits = model(input_ids)                          # 前向
    loss = cross_entropy(logits, input_ids)            # 让模型学会预测下一个词
    loss.backward()
```

这样写有几个问题？我们一条条拆。

---

### 2. 第一个编译失败：模型预测的是"下一个词"，不是"当前词"

**问题 1：因果语言模型的预测方向**

Causal LM 的 forward 输出 `logits[i]` 预测的是**第 i+1 个 token**，不是第 i 个。

```
输入序列:  [A, B, C, D]
预测目标: [B, C, D, E]    ← 每个位置预测下一个 token
```

所以 `input_ids` 和 `labels` 要错开一位：

```python
input_ids = tokens[:-1]    # 取前 N-1 个 → [A, B, C]
labels    = tokens[1:]     # 取后 N-1 个 → [B, C, D]
```

这个操作叫 **Shift**，是所有 decoder-only LM 训练的标配。不是谁设计出来的，是**因果 masking 的数学定义逼出来的**——每个位置只能看见前面的 token，那它要预测的就是下一个 token，不是自己。

**问题 2：prompt 部分不该贡献梯度**

在上面的 naive 实现里，模型会在 prompt 的每个位置也学习预测下一个词。但 prompt 是用户给的，模型不需要学习"如何生成用户的问题"。如果让模型学 prompt，它会：
- 浪费参数量在记住训练集的问题模式上
- 在推理时倾向于重复问题而不是回答问题

所以我们需要一个 **Mask**，只在 response 部分算 loss：

```
tokens:    [prompt_token_1, prompt_token_2, ..., response_token_1, response_token_2, ...]
mask:      [0,              0,              ..., 1,               1,               ...]
```

**问题 3：padding 后 mask 也要跟着 shift**

batch 内序列长度不同，需要 padding 对齐。padding 的位置 mask 也应该是 0。

---

### 3. 第一个真正的实现：tokenize_prompt_and_output

上面的三个问题（Shift、Response Mask、Padding）被同一个函数解决。

```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    输入：
        prompt_strs:  ["问题1", "问题2", ...]
        output_strs:  ["回答1", "回答2", ...]
        tokenizer:    HuggingFace tokenizer
    输出：
        {
            "input_ids":      (B, max_len-1)   ← shift 后的输入
            "labels":         (B, max_len-1)   ← shift 后的标签
            "response_mask":  (B, max_len-1)   ← 1=response, 0=prompt/padding
        }
    """
    all_input_ids = []
    all_response_masks = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        combined_ids = p_ids + o_ids
        all_input_ids.append(combined_ids)
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_response_masks.append(mask)

    # Right-padding
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, (ids, m) in enumerate(zip(all_input_ids, all_response_masks)):
        padded_input_ids[i, :len(ids)] = torch.tensor(ids)
        padded_masks[i, :len(ids)] = torch.tensor(m)

    # Shift
    return {
        "input_ids": padded_input_ids[:, :-1],
        "labels": padded_input_ids[:, 1:].clone(),
        "response_mask": padded_masks[:, 1:],
    }
```

**为什么右填充？**

SFT 训练用右填充，因为我们要模型看到完整的 prompt 后再开始预测 response。
> 训练和推理的 padding 方向不同——训练右填充，vLLM 推理左填充，是**两个场景的约束不同逼出来的不同选择**。

---

### 4. 第二个实现：masked_normalize

现在我们有了 response_mask，怎么用它算 loss？

```python
loss = F.cross_entropy(logits, labels, reduction='none')  # 每个 token 的 loss
masked_loss = loss * response_mask                          # prompt 部分置零
final_loss = masked_loss.sum() / response_mask.sum()        # 只对 response 部分平均
```

这个模式在 SFT 中反复出现：**对 tensor 做 mask，然后求和再除以一个常数做归一化**。

```python
def masked_normalize(tensor, mask, normalize_constant, dim=None):
    masked_tensor = tensor * mask
    if dim is None:
        total_sum = torch.sum(masked_tensor)
    else:
        total_sum = torch.sum(masked_tensor, dim=dim)
    return total_sum / normalize_constant
```

为什么 `normalize_constant` 是参数而不是 `mask.sum()`？在梯度累积场景中，一个 microbatch 的 loss 要除以 `grad_accum_steps`，这个除数和 mask 的 token 数无关。

---

### 5. 第三个实现：sft_microbatch_train_step

```python
def sft_microbatch_train_step(policy_log_probs, response_mask,
                               gradient_accumulation_steps):
    """
    policy_log_probs:         (B, L)   每个 token 的 log_prob
    response_mask:            (B, L)   0/1掩码
    gradient_accumulation_steps:  int

    返回:
        scaled_loss:  已除以 grad_accum_steps 的 loss（用于 backward）
        metadata:     { "loss": 未缩放的 microbatch 平均 loss }
    """
    nll_per_token = -policy_log_probs

    # 关键：用 mask.sum() 做归一化，得到每 token 平均 NLL
    num_valid_tokens = response_mask.sum().clamp(min=1)
    total_masked_loss = torch.sum(nll_per_token * response_mask)
    microbatch_loss_mean = total_masked_loss / num_valid_tokens

    # 梯度累积缩放
    scaled_loss = microbatch_loss_mean / gradient_accumulation_steps

    return scaled_loss, {"loss": microbatch_loss_mean.detach()}
```

**关键问题：为什么要除以 gradient_accumulation_steps？**

因为梯度累积的含义是：连续 N 个 microbatch 的梯度累加后再更新参数。PyTorch 的 `loss.backward()` 是**累加**梯度到 `.grad` 上。

```
微批次 1: loss_1 → loss_1/N → backward → grad = grad_1/N
微批次 2: loss_2 → loss_2/N → backward → grad = grad_1/N + grad_2/N
...
微批次 N:                            → optimizer.step() → grad = (sum grad_i)/N
```

这就是梯度累积缩放因子的由来——不是某个框架的设计，是**数学上必须保持一致**。

---

### 6. MATH 验证集评估

SFT 之前先跑 baseline。但 MATH 的答案判分和 GSM8K **完全不同**：

| 数据集 | 答案示例 | parse_last_number 结果 | 正确判断 |
|--------|---------|------------------------|----------|
| GSM8K | `#### 72` | `"72"` ✅ | 数字相等 |
| MATH | `\dfrac{1}{9}` | `"9"` ❌ | 需要 LaTeX 解析 |
| MATH | `\boxed{420}` | `"420"` ✅ 运气 | 需要 \boxed 提取 |
| MATH | `\text{4:30 p.m.}` | `"30"` ❌ | 需要 \text 解析 |

GSM8K 的 `parse_last_number("最后一个数字")` 对 MATH 完全无效。MATH 答案用 LaTeX 表示，需要**符号级等价性判断**。

解决方案：复用 `reasoning/rewards.py` 中的 `grade()` 函数：

```python
def grade(model_answer, ground_truth, fast=True):
    # 1. grade_answer_mathd: 字符串归一化后比较
    # 2. grade_answer_sympy: sympy 符号等价性判断
    # 3. (可选) is_latex_equal: math_verify 库做 LaTeX 语义比较
    correct = grade_answer_mathd(...) or grade_answer_sympy(...)
    if not fast:
        correct = correct or is_latex_equal(...)
    return correct
```

基线结果（Qwen2.5-Math-1.5B，零样本，5000 条 MATH 验证集）：

```
accuracy:  0.2254  (22.5%)
format_rate: 0.6524 (65% 的输出遵循了 <answer> 格式)
```

---

### 7. 数据蒸馏：从 MATH train 生成推理链

MATH train 有 7,500 道题（problem + answer），但没有推理链。需要让 DeepSeek API 为每道题生成 `<think>...</think> <answer>...</answer>` 格式的推理轨迹。

蒸馏流程：

```
MATH train.jsonl (7,500 题, problem + answer)
  │
  ▼  r1_zero 模板包装
  │
  调 DeepSeek API → 生成推理链
  │
  ▼  校验答案是否匹配 ground_truth（答错跳过）
  │
  保存为 SFT 格式 (prompt + response)
```

结果：
- 第一轮：2,067 正确 / 5,433 跳过（通过率 27.6%）
- 第二轮（断点续传）：3,052 正确 / 2,381 跳过（通过率 56.2%）
- **总计：5,119 条有效 SFT 数据**

通过率从 27.6% 提升到 56.2% 的原因是 DeepSeek 的 API 在连续调用中生成了更稳定的结果，也可能因为第二轮的题目相对简单（随机抽样导致的分布差异）。

注意：1.5B 的小模型答 MATH 的正确率只有 22.5%，所以用 7B+ 的 DeepSeek API 蒸馏（56% 通过率）是合理的——**学生模型不会的知识，让老师模型教**。

---

### 8. Packed SFT Dataset：用 packing 替代 padding

朴素做法：每条 `prompt + response` 独立 tokenize → padding 对齐 → 训练。

问题：短序列 padding 浪费大量 GPU 算力。如一条 200 token 的样本 padding 到 1024，80% 的计算量浪费在 pad token 上。

**Packed SFT dataset**：把所有文档拼接成一条超长 token 序列，然后按 `seq_length` 切块。

```
文档 1: [t1, t2, ..., tn, EOS]
文档 2: [t1, t2, ..., tm, EOS]
   ↓ 拼接
[t1, t2, ..., tn, EOS, t1, t2, ..., tm, EOS, ...]
   ↓ 按 1024 切块
块 0: [  0,   1, ..., 1023]  →  labels: [  1,   2, ..., 1024]
块 1: [1024, 1025, ..., 2047] →  labels: [1025, 1026, ..., 2048]
```

5,119 条蒸馏数据 → `PackedSFTDataset(seq_length=1024)` → **1,521 条 packed 序列**。减少了 70% 的 padding 浪费。

---

### 9. 完整的训练循环

```python
# 加载 packed 数据
dataset = get_packed_sft_dataset(tokenizer, data_path, seq_length=1024)

# 分批训练
for batch in iterate_batches(dataset, batch_size=32):
    # 拆分为 micro-batch（梯度累积）
    microbatches = split_microbatches(batch, grad_accum_steps=8)
    for micro_batch in microbatches:
        logits = model(micro_batch["input_ids"]).logits
        log_probs = compute_log_probs_from_logits(logits, micro_batch["labels"])
        loss, meta = sft_microbatch_train_step(log_probs, response_mask, grad_accum_steps)
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # 定期在 MATH 验证集上评估
    if step % eval_every == 0:
        eval_metrics = _run_eval(model, tokenizer, val_prompts, val_golds)
        wandb.log(eval_metrics, step=step)
```

训练到 500 步时 loss 从 ~6.4 下降到可接受范围。wandb 记录 loss、lr、eval accuracy 曲线。

---

### 10. 实际训练效果

```text
step     0/500 | loss: 6.4336 | lr: 4.00e-07 | elapsed: 5s
step    20/500 | loss: 5.9876 | lr: 8.40e-06 | elapsed: 101s
step    40/500 | loss: 4.2140 | lr: 1.64e-05 | elapsed: 202s
step    60/500 | loss: 2.8560 | lr: 1.95e-05 | elapsed: 301s
step    80/500 | loss: 2.4400 | lr: 1.86e-05 | elapsed: 403s
step   100/500 | loss: 2.2790 | lr: 1.77e-05 | elapsed: 503s
```

loss 从 6.4 降到 2.28，模型在持续学习。对比：
- 完全随机（vocab=151k）：NLL ≈ log(151k) ≈ 11.93
- step 0：6.43（比随机好，因为 Qwen 基座已经预训练过数学）
- step 100：2.28（显著下降，说明 SFT 在生效）
- 理想终值：< 1.0（模型高置信度预测正确答案）

硬件：L40 46GB，batch_size=4，grad_accum=8（等效 batch_size=32），eager attention（flash-attn 未安装）。

---

## 二、八股概念基础知识点

### 2.1 负对数似然（NLL, Negative Log-Likelihood）

**决策链位置**：`sft_microbatch_train_step` 中的 `-policy_log_probs`

对于 ground truth token `y`，模型预测它的概率是 `p(y)`。NLL = `-log p(y)`。
- 90% 预测对 → `-log(0.9) = 0.105`，小
- 10% 预测对 → `-log(0.1) = 2.302`，大

### 2.2 梯度累积（Gradient Accumulation）

**决策链位置**：`scaled_loss = microbatch_loss_mean / gradient_accumulation_steps`

当显存不足以容纳大 batch 时，将大 batch 拆成 N 个 micro-batch，逐个前向+反向，梯度累加 N 次后 optimizer.step()，等效于大 batch 训练。

### 2.3 Loss 归一化：SUM 还是 MEAN

**这是本节最重要的考点。**

SFT loss 有两种归一化策略：

| 策略 | 计算方式 | 效果 | 什么时候用 |
|------|---------|------|-----------|
| **Token-level MEAN** | `sum(nll * mask) / sum(mask)` | 每 token 平均 NLL | 标准 SFT |
| **Sequence-level MEAN** | `mean( sum(nll_i) / len_i )` | 每条样本等权重 | 长文本均匀贡献 |

**实战教训**：最初写的是 `sum(nll) / 1.0`（等价于没归一化），loss 达到 6588——这是一个总和，完全不可读。必须用 `mask.sum()` 做归一化，得到每 token 平均 NLL，才能在跨实验间比较。

### 2.4 Shift 操作

```
tokens = [A, B, C, D, E]
input_ids = [A, B, C, D]    ← 前 N-1 个
labels    = [B, C, D, E]    ← 后 N-1 个
```

因果语言模型的**定义约束**：每个位置只能预测下一个位置。

### 2.5 Padding Side 选择

| 场景 | Padding 方向 | 原因 |
|------|-------------|------|
| SFT 训练 | **右填充** | 模型需要看到完整的 prompt 后再开始生成 |
| vLLM 推理 | **左填充** | 保证所有序列的最后一个 token 对齐，便于并行 decode |

L40 实际限制：一次 decode 最多 8 条，否则显存不足。

### 2.6 Entropy 监控

SFT 训练中监控 entropy 可以发现问题：
- entropy 过低 → 模型只在少数 token 上有高概率，可能过拟合
- entropy 过高 → 模型对所有 token 一视同仁，还没学到有效模式

### 2.7 Packed SFT Dataset

Packing 的核心思想：**用 EOS 分隔符把多篇文档拼成一条超长序列，按固定长度切块**。

优点：
- 几乎零 padding，GPU 利用率接近 100%
- 无需动态 batching，输出张量形状固定，stack 开销极低

代价：
- 文档边界被打散，一条 packed 序列可能包含多篇文档片段
- 无法用 response_mask 区分 prompt 和 response 区域（所有 token 都是训练目标）

对于蒸馏数据（prompt + response 已格式化），所有 token 都是有效的训练目标，所以 packing 的代价可以忽略。

### 2.8 MATH 判分的 LaTeX 等价性

MATH 答案的判分需要三阶段：

1. **字符串归一化**：去空格、去 `\boxed{}`、统一 `\dfrac` → `\frac`
2. **sympy 符号化简**：`simplify(expr1 - expr2) == 0` 判断数学等价
3. **兜底 math_verify**：LaTeX 语义解析 + 符号比较

这三个阶段不是预先设计的，而是被不同类型的 MATH 答案逼出来的：
- 纯数字 42 → 阶段 1 就够了
- `\frac{1}{9}` vs `1/9` → 必须阶段 2
- `\boxed{420}` vs `420` → 阶段 1 + `\boxed` 提取
- `\text{4:30 p.m.}` → 字符串匹配即可

---

## 三、排障过程实践

### 现象 1：Loss 值 6588，完全不可读

```text
step 0/500 | loss: 6588.0000
```

**根因**：`sft_microbatch_train_step` 中 `normalize_constant=1.0`，loss = `sum(nll * mask) / 1.0` = NLL 的总和。对 B=4, L=1024，每个 token NLL ≈ 1-12，总和 ≈ 4×1024×1.6 ≈ 6553。

**修复**：用 `response_mask.sum()` 替代 1.0 做归一化，得到每 token 平均 NLL。修正后 step 0 的 loss 为 6.43，合理。

**教训**：SFT loss 一定要按有效 token 数平均（MEAN），不能只求和（SUM）。6588 是 SUM 的典型值，完全不可跨实验比较。

### 现象 2：Right-padding detected 刷屏

```text
[transformers] A decoder-only architecture is being used, but right-padding was detected!
```

**根因**：`_run_eval` 中 HF generate 时 tokenizer 处在训练模式（右填充），但生成需要左填充。训练时 `padding_side = "right"`（SFT 训练的标准），eval 时忘记改回来。

**修复**：eval 生成前设 `tokenizer.padding_side = "left"`，生成完恢复。

### 现象 3：Packed 数据集 tensor 形状不一致

```text
RuntimeError: stack expects each tensor to be equal size, but got [1024] at entry 0 and [1023] at entry 28
```

**根因**：`PackedSFTDataset` 切块逻辑中，最后一个 `chunk` 不足 `seq_length + 1` 时，`label_chunk = chunk[1:seq_length + 1]` 比 `input_chunk = chunk[:seq_length]` 短 1。

**修复**：改成 while 循环逐块切分，最后一个块独立处理 padding：`input_chunk + [pad_id] * pad_len`，`label_chunk = last_valid_tokens + [pad_id] * (pad_len + 1)`。

### 现象 4：flash-attn 未安装导致 OOM

```text
WARNING: flash-attn not installed, falling back to 'eager' attention
```

L40 46GB 在不装 flash-attention 时，batch_size=64 直接 OOM。只能降到 batch_size=4，grad_accum=8（等效 32）。

**修复**：安装 flash-attention：
```bash
uv pip install flash-attn --no-build-isolation
```
装好后 batch_size=64 应能正常工作。

### 现象 5：MATH 蒸馏数据被 Alpaca 模板双重包装

原始蒸馏数据已经用 r1_zero 模板格式化过 prompt，但 `PackedSFTDataset` 又包了一层 Alpaca 模板：

```python
# ❌ 双重包装
text = "Below is an instruction...\n### Instruction:\n" + prompt + "\n\n### Response:\n" + response
# prompt 本身已经是 "A conversation between User and Assistant...\nUser: ...\nAssistant: <think>"
```

这导致训练数据变成 "Alpaca 包 r1_zero 包真实内容" 的三层嵌套，模型学的是"如何嵌套模板"而不是"如何回答数学题"。

**修复**：直接 `text = prompt + response`，不做额外包装。

### 现象 6：L40 显存限制与 batch size 选择

| 配置 | 显存占用 | 能否运行 |
|------|---------|---------|
| batch_size=64, eager | OOM | ❌ |
| batch_size=8, eager | 约 28GB | ✅ |
| batch_size=4, grad_accum=8, eager | 约 18GB | ✅ |
| batch_size=64, flash-attn | 约 20GB | ✅（预期） |

1.5B 模型权重约 3GB（bfloat16），显存大头在激活值（activations）。Flash attention 通过分块计算大幅降低显存占用。

### 现象 7：蒸馏通过率波动

- 第一轮 7,500 题：2,067 正确（27.6%）
- 第二轮断点续传 5,433 题：3,052 正确（56.2%）

通过率差异源于 DeepSeek API 的随机性（temperature=0.7）和题目难度分布差异。总蒸馏有效数据 5,119 条。

### 现象 8：Loss 归一化错误导致等效学习率放大 1024 倍

训练到 200 步时 eval accuracy 从 65% 回退到 64%，模型开始抖动。

**根因**：旧版 `sft_microbatch_train_step` 中 `normalize_constant=1.0`，等效于 loss = `sum(-log_prob) / batch_size / grad_accum_steps`，没有除以有效 token 数。

对比新旧两版的 loss 计算：

```
旧版: scaled_loss = sum(-log_prob) / 1.0 / batch_size / grad_accum_steps
新版: scaled_loss = sum(-log_prob) / mask.sum() / grad_accum_steps

对新版来说，mask.sum() ≈ batch_size × seq_length = 4 × 1024 = 4096
所以旧版的梯度是新版的 4096 / 1 = 4096 倍

但旧版还多除了一个 batch_size（4），所以实际差：
旧版梯度大小 = sum / 4 / 8 = sum / 32
新版梯度大小 = sum / 4096 / 8 = sum / 32768
旧版 / 新版 = (sum/32) / (sum/32768) = 1024 倍
```

**等效学习率**：旧版 `lr=2e-5` 实际生效为 `2e-5 × 1024 ≈ 2%`。正常的 SFT 学习率在 1e-5 到 5e-5 之间，2% 大了 1000 倍。这解释了：

- 为什么 100 步就冲到 65%（学习率巨大，参数一步跨很远）
- 为什么 200 步开始抖动（学习率太大，参数在最优值附近震荡，无法收敛到极小值点）

**修复**：去掉 `normalize_constant` 参数，直接用 `mask.sum()` 做归一化，得到每 token 平均 NLL。修正后 step 0 的 loss 从 6588（总和）变成 6.43（平均值）。

**教训**：SFT loss 的归一化方向决定了有效学习率的尺度。`sum / token_count` 才是正确的每 token 平均 NLL。用 `sum / 1.0` 等于把梯度放大了一个 `token_count` 倍的常数因子，等效学习率失控。

### 现象 9：tok/s 显示随训练步数递减

控制台输出的 tok/s 从 step 20 的 42 一路降到 step 220 的 3。

**根因**：tok/s 的计算除的是 `time.perf_counter() - start_time`（训练开始以来的总耗时），不是当前步的耗时。随着总耗时不断累加，tok/s 必然单调递减。

```python
# ❌ 错误
tokens_per_sec = (batch_size * seq_length) / (time.perf_counter() - start_time)

# ✅ 正确：跟踪每步耗时
step_start_time = time.perf_counter()
# ... 训练 ...
step_time = time.perf_counter() - step_start_time
tokens_per_sec = tokens_per_step / step_time
```

**修复**：添加 `step_start_time` 变量，每步更新，计算当前步的瞬时吞吐率。

## 面试表达

```
我不是一开始就想好了要有 tokenize_prompt_and_output、masked_normalize、
sft_microbatch_train_step、PackedSFTDataset 这些函数。

我是先写了一个最简单的循环：读数据 → 拼起来 → 算 loss → backward。

然后遇到了问题：
1. 因果 LM 要求输入和标签错开一位（Shift）
2. 模型不该学如何生成 prompt（Response Mask）
3. 序列长度不同需要 padding
这三个问题把 tokenize_prompt_and_output 逼了出来。

然后发现 mask → sum → normalize 的模式在 loss 计算和评估时重复出现，
所以 masked_normalize 被抽了出来。

再然后发现单步 loss 计算在梯度累积时需要缩放，
所以 sft_microbatch_train_step 承担了这个逻辑。

接下来发现 5,119 条 SFT 数据每条长度不同，padding 浪费严重，
所以 PackedSFTDataset 把文档拼接后按固定长度切块。

最后发现 loss 值 6588 完全不可读——因为 normalize_constant=1.0 等于没归一化，
改成 mask.sum() 后 loss 变成 6.43，这才是正常的每 token 平均 NLL。

所有函数都是为了解决一个具体的编译失败 / 训练异常 / 评估需求出现的，
不是为了凑某个设计模式。
```
