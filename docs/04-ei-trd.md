# Expert Iteration：从"学老师的"到"学自己的"

## 开篇：Expert Iteration 在做什么？

**Expert Iteration（EI）** 的流程是：

> 用当前策略模型自己生成多条推理轨迹 → 筛选出正确的 → 用这些正确的轨迹做 SFT → 重复。

和 SFT 的核心区别：

| | SFT | Expert Iteration |
|---|---|---|
| 训练数据来源 | 外部老师（DeepSeek API 蒸馏） | **自己生成的正确轨迹** |
| 上限 | 老师的水平（56% 通过率） | **自己探索的上限（无上限）** |
| 迭代 | 一次性 | **多轮迭代，越滚越好** |

EI 不是 SFT 的替代，而是 **接力**：SFT 先把模型从 22.5% 拉到 54.5%，EI 再自己探索往 80%+ 走。

---

## 一、MVP 手把手最小实现

### 1. 第一个愿望：模型在 SFT 之后怎么继续提升？

SFT 100 步后，MATH 从 22.5% 跳到 54.5%，然后卡住了。

为什么卡住？因为蒸馏数据是 **DeepSeek API 生成的**。API 在 MATH 上的通过率只有 56%，所以蒸馏数据里：
- 56% 是正确的推理链（被保留）
- 44% 是错误的推理链（被丢弃）
- 蒸馏数据里**没有老师也不会做的题**

SFT 的上限就是老师的上限——这是**模仿学习（Behavioral Cloning）** 的固有问题：学生不能超过老师。

所以需要一种方法，让模型**自己生成数据、自己筛选、自己学习**。这就是 Expert Iteration。

### 2. 第一次尝试：用当前模型生成 → SFT

最朴素的想法：

```python
for round in range(ei_rounds):
    # Step 1: 用当前模型生成多条回答
    responses = []
    for question in dataset:
        for _ in range(N):
            response = model.generate(question)
            responses.append((question, response))

    # Step 2: 筛选正确的
    correct_data = []
    for question, response in responses:
        answer = extract_answer(response)
        if answer and grade(answer, ground_truth):
            correct_data.append({"prompt": question, "response": response})

    # Step 3: SFT 训练
    sft_train(correct_data)
```

但有几个问题：

**问题 1：模型初期生成的正确轨迹很少**

Round 1 时模型只有 54.5% 准确率，每个问题生成 8 条，期望正确数 = 8 × 54.5% ≈ 4.36 条。但有些问题模型全错（一条正确的都没有），这些问题的正确解法学不到。

**问题 2：正确轨迹里有重复模式**

模型倾向于用类似的推理路径。如果 8 条里有 3 条正确，它们可能用了相同的解法。这导致数据多样性不足。

**问题 3：错误的轨迹可能包含正确的局部推理**

有时模型推理的前半段是对的，最后算错了。EI 的 filter 会丢弃整条轨迹，浪费了前半段正确的推理步骤。

不过这些问题在实际中影响不大——EI 多轮迭代后，模型准确率逐步提升，正确轨迹的比例也会逐步提升。

### 3. EI 的核心循环

```python
def expert_iteration_round(
    model, dataset, reward_fn, n_generations=8, eval_fn=None
):
    """
    一轮 Expert Iteration：
    1. 对每条问题生成 n_generations 条回答
    2. 用 reward_fn 筛选正确的
    3. 对正确的回答做 SFT
    """
    # ── Phase 1: Generate rollouts ──
    # 用 vLLM 批量生成，速度快
    all_prompts = []
    all_responses = []
    all_golds = []

    for example in dataset:
        prompt = make_math_prompt(example["problem"])
        gt = example["answer"]
        for _ in range(n_generations):
            all_prompts.append(prompt)
            all_golds.append(gt)

    # vLLM 批量生成
    outputs = vllm_model.generate(all_prompts, sampling_params)
    all_responses = [output.outputs[0].text for output in outputs]

    # ── Phase 2: Filter correct trajectories ──
    sft_data = []
    for prompt, response, gt in zip(all_prompts, all_responses, all_golds):
        scores = reward_fn(response, gt)
        if scores.get("reward", 0.0) > 0.5:  # 完全正确（format + answer）
            sft_data.append({
                "prompt": prompt,
                "response": response,
            })

    print(f"  Correct: {len(sft_data)} / {len(all_prompts)} "
          f"({len(sft_data)/len(all_prompts):.1%})")

    # ── Phase 3: SFT on filtered data ──
    # 复用已有的 SFT 训练逻辑
    packed_dataset = get_packed_sft_dataset(tokenizer, sft_data, seq_length=1024)
    for step in range(sft_steps_per_round):
        batch = next(iter(iterate_batches(packed_dataset, batch_size=32)))
        loss = sft_train_step(model, batch)
        loss.backward()
        optimizer.step()

    # ── Phase 4: Evaluate ──
    if eval_fn:
        eval_fn(model)

    return model, sft_data
```

### 4. 为什么 EI 能超过老师？

EI 的核心洞察：**每轮正确的轨迹是由当前的策略模型生成的。模型变好了，生成的正确轨迹就变多了；正确轨迹变多了，SFT 后模型就变得更好。**

```
Round 1: 模型 accuracy 54.5% → 生成 8 条 × 1000 题 → ~4,360 条正确轨迹
  → SFT → 模型 accuracy 60%
Round 2: 模型 accuracy 60% → 生成 8 条 → ~4,800 条正确轨迹
  → SFT → 模型 accuracy 65%
Round 3: 模型 accuracy 65% → 生成 8 条 → ~5,200 条正确轨迹
  → SFT → 模型 accuracy 68%
...
```

这是一个**正反馈循环（virtuous cycle）**——不像 SFT 受限于外部老师，EI 可以一直迭代到模型自身的 expressiveness 上限。

### 5. 和蒸馏的区别

| | DeepSeek API 蒸馏 | Expert Iteration |
|---|---|---|
| 老师 | DeepSeek（大模型） | **自己（小模型）** |
| 通过率 | 56% | 54.5% → 逐步提升 |
| 推理风格 | DeepSeek 的风格 | **自己的风格** |
| 成本 | API 调用费（耗时） | **免费（自己生成）** |
| 迭代 | 一次性 | **多轮** |

有趣的是，EI 的正确轨迹数量可能一开始不如蒸馏（54.5% vs 56%），但 EI 的轨迹是**模型自己风格的**——模型学自己的推理路径比学 DeepSeek 的更容易，因为格式、措辞、推理步长都和自己一致。

---

## 二、八股概念基础知识点

### 2.1 Expert Iteration vs SFT vs RL

| 方法 | 数据来源 | 奖励信号 | 更新方式 |
|------|---------|---------|---------|
| SFT | 外部老师 | 隐式（老师的回答就是对的） | 一次性 |
| **Expert Iteration** | **自己生成+筛选** | **答案对错（filter）** | **迭代** |
| GRPO/RL | 自己生成 | 连续奖励 + 组归一化 | 策略梯度 |

EI 是 SFT 和 GRPO 之间的**中间步骤**：
- 比 SFT 多了一个"自己生成"的环节
- 比 GRPO 简单——不需要策略梯度、不需要 KL 散度、不需要 advantage 归一化

### 2.2 正反馈循环（Virtuous Cycle）

EI 的核心机制：

```
更好的策略 → 生成更多正确轨迹 → 更多训练数据 → 更好的策略
```

这个循环依赖一个条件：**模型至少能偶尔生成正确的轨迹**。如果模型准确率为 0%，EI 无效。SFT 的作用就是把模型拉到足够高的起点（~50%+），让正反馈循环转起来。

### 2.3 生成数量（N）的选择

每个问题生成 N 条回答，N 的选择影响：

| N | 优点 | 缺点 |
|---|------|------|
| 小（2-4） | 生成快，显存占用小 | 正确轨迹少，尤其是难题 |
| 大（8-16） | 难题也有概率蒙对一条 | 生成慢，数据重复率高 |

实际选择取决于：
- 模型当前准确率：低 → 需要大 N 才能采到正确轨迹
- 推理速度：vLLM 下 N=8 通常可接受
- 显存：batch_size × N 不能超限

### 2.4 SFT Steps Per Round（SPR）

EI 每轮在筛选后的数据上做 SFT，但做多少步？

- SPR 太小：模型没学充分，提升有限
- SPR 太大：模型过拟合到当前的正确轨迹分布，失去探索能力

经验值：SPR 通常为几十到几百步，取决于数据量。

### 2.5 灾难性遗忘（Catastrophic Forgetting）

SFT 在 MATH 上从 22.5% → 54.5%，但 GSM8K 从 70.7% 掉到 57.5%。

EI 同样会面临这个问题——如果只在 MATH 上迭代，GSM8K 会继续掉。缓解方法：
- **Replay**：每轮混合一部分 GSM8K 数据
- **Multi-task**：同时优化多个 benchmark
- **EWC（Elastic Weight Consolidation）**：对重要参数加正则

### Interview phrasing

> Expert Iteration 是在 SFT 和 GRPO 之间的过渡方法。SFT 受限于外部老师的水平——老师不会做的题，学生永远学不到。EI 让模型自己生成多条推理路径，筛选出正确的做 SFT，下一轮再用更好的模型生成更多正确路径。这个正反馈循环让模型可以突破老师的上限。和 GRPO 相比，EI 不需要策略梯度、advantage 归一化这些复杂机制，但收敛速度和对困难样本的覆盖不如 GRPO。

---

## 三、排障过程实践

### 现象 1：EI 第一轮的正确轨迹数量远低于蒸馏

| 数据来源 | 正确率 | 5,000 题 × 8 条的期望正确数 |
|---------|--------|---------------------------|
| DeepSeek 蒸馏 | 56% | —（一次性 7,500 题） |
| EI Round 1 | 54.5% | 5,000 × 8 × 54.5% = **21,800 条** |
| EI Round 5 | 68% | 5,000 × 8 × 68% = **27,200 条** |

即使第一轮，EI 的期望正确轨迹数也远超蒸馏——因为生成了更多候选（8 条/题 vs 蒸馏的 1 条/题）。这是 EI 的一个隐含优势：**通过多次采样来弥补单次准确率的不足**。

### 现象 2：vLLM rollouts 需要 sync 权重

EI 的生成阶段用 vLLM（快），SFT 阶段用 HF（需要梯度）。每轮需要：
1. 把训练完的 HF 模型权重 sync 到 vLLM
2. 用 vLLM 批量生成
3. 收集结果，筛选，SFT

权重同步可以用已有的 `vllm_utils.py` 中的 `sync_policy_weights`，需要两卡：cuda:0 训练，cuda:1 跑 vLLM server。

如果没有两卡，可以用 HF 的 `model.generate()` 代替——慢一些但能跑。

### 现象 3：筛选标准的选择

`r1_zero_reward_fn` 返回三个分数：

```python
{
    "format_reward": 1.0,    # 有 <think> + <answer> 标签
    "answer_reward": 1.0,    # 答案正确
    "reward": 1.0,           # 两者都满足
}
```

筛选时可以选择：
- **严格模式**：`reward > 0.5` → 格式和答案都要对
- **宽松模式**：`answer_reward > 0.5` → 答案对了就行，格式不对也保留
- **格式优先**：`format_reward > 0.5` → 至少格式要对，答案可以错（保留推理过程）

对于 EI，推荐**严格模式**——只保留完全正确的轨迹，避免模型学到错误推理。

### 现象 4：数据重复与过拟合

EI 每轮生成的正确轨迹中，同一个问题的多条正确回答可能高度相似（模型倾向于用相同的推理路径）。这会导致：
- 数据多样性不足
- 模型过拟合到少数推理模式

缓解方法：
- 生成时 temperature > 0（如 0.7-1.0），增加多样性
- 每轮去重：对同一问题的多条正确轨迹，只保留 1-2 条
- 混合历史数据：把前几轮的正确轨迹也混入训练

### 现象 5：首轮 EI 获得 0% 正确轨迹

**表现**：temperature=0.7、generations=2、20 题，一轮下来 0/40 正确。

**根因**：采样温度 > 0 会摊平概率分布。对于一个贪心准确率 55% 的模型，temperature=0.7 下正确 + 格式合规的概率远低于 55%。每个问题只生成 2 条，20 题总共 40 条里一条正确的都没采到。

**修复**：首轮用 `--ei_temperature 0.0`（贪心解码），确保采到正确轨迹。后续轮次再逐步提高温度增加多样性。

**经验规律**：
- Round 1: temperature=0.0，最大化正确轨迹数
- Round 2: temperature=0.3，在保证质量的前提下增加多样性
- Round 3+: temperature=0.5-0.7，模型变强了，可以承受更高的采样风险

### 现象 6：Rollout 多样性极差——93%+ 的 response 完全相同

**问题表现**：用 `scripts/check_rollout_diversity.py` 检查每个 prompt 的 4 条 response，结果显示：

| 轮次 | 4 条全同 | 有差异 | 差异占比 |
|------|---------|-------|---------|
| Round 1 | 3,847 | 261 | 6.4% |
| Round 2 | 4,214 | 128 | 2.9% |
| Round 3 | 4,426 | 90 | 2.0% |

有差异的 prompt 平均编辑距离 **0.0 字符**，说明差异极小（多为 token 概率相等时的浮点非确定性断点）。

但明明设置的是 `--ei_temperature 0.7`，为什么实际生成出来的结果像贪心解码？

**排障过程**：

1. **第一步：确认温度传递链路**
   ```
   CLI args.ei_temperature (0.7)
     → expert_iteration_round(temperature=0.7)      ← ✅ 传到了
         → vllm_generator.generate(all_prompts, max_new_tokens=...)  ← ❌ temperature 没传
   ```

2. **第二步：检查 `VLLMGenerationConfig.generate()` 接口**
   ```python
   # cs336_alignment/eval/generation.py 第 183-189 行
   def generate(
       self,
       prompts: list[str],
       *,
       max_new_tokens: int,
       stop_strings: list[str] | None = None,   # ← 没有 temperature 参数
   ) -> list[str]:
   ```
   接口签名根本没有 `temperature` 参数，所以 `expert_iteration_round` 的 `temperature=0.7` 传不进来。

3. **第三步：检查 `SamplingParams` 构造**
   ```python
   sampling_params = SamplingParams(
       max_tokens=max_new_tokens,
       temperature=0.0,   # ← 硬编码 0.0！
       top_p=1.0,
       stop=stop_strings,
   )
   ```
   `temperature` 被写死为 0.0，无论调用者传什么都没用。

4. **根因定位**

   文件注释中已经写了原因（第 10-12 行）：
   ```python
   # 之后 GRPO rollout 也可以复用这里的 vLLMGenerator；
   # 现在先只做 greedy / temperature=0 的确定性生成。
   ```
   `VLLMGenerationConfig` 最初只服务于 baseline eval，设计意图就是贪心解码。后来 EI 直接复用了它做 rollout 生成，但**忘了给它加 temperature 参数**。

   因此，整个 EI 实验实际上一直在用 `temperature=0.0` 做 rollout——所有 4 条 response 都由贪心解码生成，自然一模一样。

5. **修复方案**（两处改动）

   **① `generation.py` — 给 generate 方法加 `temperature` 参数**
   ```python
   def generate(
       self,
       prompts: list[str],
       *,
       max_new_tokens: int,
       temperature: float = 0.0,      # ← 新增，默认 0.0 保持向后兼容
       stop_strings: list[str] | None = None,
   ) -> list[str]:
       sampling_params = SamplingParams(
           max_tokens=max_new_tokens,
           temperature=temperature,     # ← 使用传入的参数
           top_p=1.0,
           stop=stop_strings,
       )
   ```

   **② `ei.py` — 调用时传入 temperature**
   ```python
   all_responses = vllm_generator.generate(
       all_prompts,
       max_new_tokens=max_new_tokens,
       temperature=temperature,
   )
   ```

6. **修复后的预期**：`temperature=0.7` 的 rollout 应该产生多样化的推理轨迹，同一 prompt 的 4 条 response 不再雷同；GRPO 和 EI 的 advantage/筛选机制才能真正发挥作用。

**思考题：那为什么 validation accuracy 还是从 54.5% 涨到了 70%？**

这个问题很好。答案分两层：

**第一，测量的是 validation 集，不是 training 集。**

代码中 `_ei_eval`（`ei.py` 第 553 行）用的是 `val_examples`（MATH validation set），且 eval 本身用 `do_sample=False` 贪心解码，所以测量是真实可靠的。

**第二，即使 rollout 全部相同，SFT 阶段仍然有效。**

EI 的流程是：Rollout → Filter（保留正确的） → SFT。即使某个 prompt 的 4 条 response 一模一样，只要它通过了 filter（答案正确），SFT 就会用这条正确轨迹训练模型。这实际上是一个 **self-distillation** 过程：

```
模型在问题上生成一条正确轨迹（虽然只有1种写法）
→ SFT 强化这个推理模式
→ 模型在该问题上更自信、更稳定
```

Qwen2.5-Math-1.5B 的基线正确率已经有 ~55%，首轮贪心生成 4 条 × 7500 题 = 30,000 条，其中约 16,620 条正确（55.4%）。SFT 在这些数据上训练一轮，模型学会稳定输出自己已掌握的正确推理模式——所以 validation 从 54.5% 涨到 69.5%。

但这个提升是**虚假的繁荣**——它来自 self-distillation 而不是来自探索新推理路径。真正的 "Expert Iteration" 应该让模型探索多种解法，筛选出正确的，然后从多样性中学到更鲁棒的推理能力。如果 rollout 没有多样性，EI 就退化成了"SFT on self-generated correct data"，探索收益归零。

这也是为什么 Round 2 → 3 的 Eval Acc 停在 70% 不再上涨——贪心解码下模型已经"榨干"了自己已掌握的推理模式，没有多样性就探索不到新的解题方式。

修复 temperature 后，期望看到的改善：
- 同 prompt 的 4 条 response 不再相同
- 部分"只有贪心解码能对、采样就错"的题可能短期掉点
- 但多轮后模型能看到更多推理路径，最终提升更大（有望突破 70% 上限）

### 完整 EI 流程

```bash
# Step 0: 将 SFT checkpoint 转为 HF 模型目录
# SFT 产出的 .pt 文件是 state_dict，不能直接被 vLLM / HF from_pretrained 加载。
# 需要先加载 base model，再加载 checkpoint 权重，然后 save_pretrained。
uv run python -c "
from transformers import AutoModelForCausalLM
import torch

# 加载 base model（结构）
model = AutoModelForCausalLM.from_pretrained(
    '/root/gpufree-share/models/Qwen2.5-Math-1.5B',
    torch_dtype=torch.bfloat16,
)

# 加载 SFT checkpoint（权重）
state = torch.load('outputs/sft_reasoning_v2/checkpoint_100.pt',
                   map_location='cpu', weights_only=True)
model.load_state_dict(state['model_state_dict'])

# 保存为完整 HF 模型目录
model.save_pretrained('/root/gpufree-share/models/Qwen2.5-Math-1.5B-SFT-step100')
print('Done')
"

# Step 1: 首轮 EI（贪心，保证产出正确轨迹）
CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_alignment.run_expert_iteration \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B-SFT-step100 \
    --device cuda:0 \
    --data_path /root/gpufree-share/data/MATH/train.jsonl \
    --val_path /root/gpufree-share/data/MATH/validation.jsonl \
    --ei_rounds 3 \
    --ei_generations_per_prompt 4 \
    --ei_sft_epochs 1 \
    --ei_batch_size 4 \
    --ei_temperature 0.0 \
    --eval_limit 200 \
    --output_dir outputs/expert_iteration_v1

# 快速烟雾测试（验证跑通）
CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_alignment.run_expert_iteration \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B-SFT-step100 \
    --device cuda:0 \
    --ei_rounds 1 \
    --ei_generations_per_prompt 2 \
    --ei_train_limit 20 \
    --eval_limit 50 \
    --ei_temperature 0.0 \
    --output_dir outputs/ei_smoke

# 每轮结束后在完整 MATH 验证集上评估
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B-EI-R1 \
    --engine vllm \
    --benchmarks math \
    --math_path /root/gpufree-share/data/MATH/validation.jsonl \
    --output_dir outputs/ei_round1_eval \
    --max_new_tokens 1024
```

---

## EI 实战结果：SFT 54.5% → EI 70.0%

**实验配置**：
- 基座：Qwen2.5-Math-1.5B（SFT Step 100 checkpoint，MATH 54.5%）
- GPU：2×L40（cuda:0 HF 训练，cuda:1 vLLM 生成）
- 3 轮，每轮 7,500 题 × 4 条 = 30,000 次生成
- 贪心解码（temperature=0.0），vLLM 耗时约 7 分/轮
- 全程 tracked on [wandb](https://wandb.ai/bbw486502970-zhejiang-university/cs336-ei-reasoning/runs/xoxeqo6s)

### 逐轮结果

| 轮次 | Rollout Acc | 正确轨迹 | Eval Acc | 耗时 | 
|------|------------|---------|---------|------|
| Baseline (SFT) | — | 5,119 (蒸馏) | **54.5%** | — |
| **EI Round 1** | 55.4% | 16,620 ✅ | **69.5%** | 24.7min |
| **EI Round 2** | 58.2% | 17,460 | **70.0%** | 27.0min |
| **EI Round 3** | 60.5% | 18,138 | **70.0%** | 26.5min |
| **总提升** | — | 52,218 条 | **+15.5pp** | 78.5min |

### 关键发现

**1. 首轮提升最大（+15pp），后续饱和**

Round 1 从 54.5% 跳到 69.5%，但 Round 2/3 只涨了 0.5pp 就到 70% 了。这说明：
- EI 的**主要收益在第一轮**——模型从"学老师的"切换到"学自己的"，格式适配带来的提升立竿见影
- 70% 可能是 **1.5B 模型用 EI 能达到的上限**（至少对 MATH 来说）

**2. 自己教自己比老师教更高效**

| 数据来源 | 正确轨迹数 | 成本 | 提升效果 |
|---------|-----------|------|---------|
| DeepSeek 蒸馏 | 5,119 | API 调用费 | 22.5% → 54.5% (+32pp) |
| EI Round 1 | 16,620 | ~25 分钟 GPU | 54.5% → 69.5% (+15pp) |

EI 的每单位正确轨迹的"学习效率"更高——因为轨迹是模型自己生成的，格式、措辞、推理步长完全一致，SFT 时 loss 下降更快。

**3. Rollout accuracy 逐轮提升**

| 轮次 | Rollout Acc | 说明 |
|------|------------|------|
| Round 1 | 55.4% | ≈ 初始 SFT 准确率 |
| Round 2 | 58.2% | +2.8pp，模型变好了 |
| Round 3 | 60.5% | +2.3pp，持续改善 |

Rollout accuracy 的提升意味着正反馈循环在生效——更好的模型 → 更多正确轨迹 → 更好的模型。

**4. Round 2→3 的 Eval Acc 停滞**

70% 之后不再提升，原因可能是：
- **1.5B 模型的 capacity 瓶颈**——参数量限制了能学会的推理模式数量
- **贪心解码限制了多样性**——temperature=0.0 下每条生成的轨迹都相似，数据多样性不够
- **灾难性遗忘**——EI 只强化了模型已经会的题（正确轨迹），不会的题依然不会

### 下一步

```
SFT (22.5% → 54.5%) → EI (54.5% → 70.0%) → GRPO (70.0% → ?)
```

EI 之后，70% 往上的收益需要 GRPO 来拿。GRPO 和 EI 的关键区别：
- **EI**：只保留完全正确的轨迹，丢弃错误的
- **GRPO**：用组归一化奖励，所有轨迹（包括错的）都贡献梯度，通过 advantage 区分好坏

GRPO 可以学到"错的没那么离谱"的推理路径中的有用信息——比如推理前半段是对的、最后算错了——而 EI 直接丢弃了这条轨迹。

---

## 面试表达

```
Expert Iteration 不是 GRPO 的简化版，而是解决了一个不同的问题。

SFT 的问题是数据来自外部老师——老师不会的题，学生永远学不到。
Expert Iteration 让模型自己生成正确轨迹来训练自己。

它比 SFT 多了一个"生成 + 筛选"环节：
1. 用当前策略生成多条推理路径
2. 筛选出答案正确的
3. 在正确路径上做 SFT

这样形成了一个正反馈循环：
正确率越高 → 能生成的正确轨迹越多 → 训练数据越丰富 → 正确率更高

和 GRPO 相比，EI 不需要策略梯度算法，
只用 SFT 就能持续提升。但它对难题的覆盖不如 GRPO——
因为 GRPO 可以用组归一化奖励来学习"错的没那么离谱"的轨迹，
而 EI 只保留完全正确的，浪费了部分正确的推理过程。
```

---

### 现象 7：Wandb 日志只有 3 个点，看不到训练曲线

**问题表现**：打开 wandb 看 EI 实验，发现 `eval_accuracy` 只有 3 个点（对应 3 轮），`train_loss` 完全没有。曲线是断开的散点，看不到模型在 SFT 过程中的变化。

**排障过程**：

1. **第一步：检查 wandb.log 调用位置**

  wandb.log 只在 `expert_iteration_loop` 里调用了一次（每轮末尾）：
  ```python
  # 旧代码只在这里 log ↓
  wandb.log({
      "eval_accuracy": eval_metrics["accuracy"],
      "rollout_accuracy": rollout_accuracy,
      ...
  }, step=round_idx)  # step=1,2,3 → 只有3个点
  ```

  而 SFT 训练阶段（`expert_iteration_round` 内部）完全没有 wandb 记录。

2. **第二步：对比参考实现**

  参考实现（`cs336-a5-RL/expert_iteration.py`）的做法：
  ```python
  # 参考实现：每个 SFT microstep 都 log
  wandb.log({"train/train_loss": loss_microbatch, "train/avg_entropy": avg_entropy},
            step=wandb_step)
  wandb_step += 1

  # 每个 SFT step 后 eval 并 log
  format_acc, answer_acc = model_eval(...)
  wandb.log({"eval/format_acc": format_acc, "eval/answer_acc": answer_acc})
  ```

  关键区别：
  - **参考实现：** SFT 阶段每步都 log，使用全局 `wandb_step` 计数器
  - **当前实现：** SFT 阶段零 log，只在轮末 log 一次

3. **第三步：根因**

  `expert_iteration_round` 内部虽然有 SFT loss 计算，但没有任何 wandb 调用。所有 wandb 日志都集中在 `expert_iteration_loop` 的轮末，而且 `step=round_idx` 把 3 轮映射到 step 1/2/3 → wandb 认为只有 3 个时间步。

4. **修复方案**（两处改动）

  **① 给 `expert_iteration_round` 传入 wandb_step 计数器**
  ```python
  def expert_iteration_round(
      ..., wandb_step: list[int] | None = None,  # mutable list 作为引用传参
  ):
  ```

  **② SFT 循环内每步 optimizer step 后 log training loss + entropy**
  ```python
  for batch in dataloader:
      # ... forward + backward ...
      optimizer.step()
      optimizer.zero_grad()

      if wandb_step is not None:
          wandb.log({
              "train/sft_loss": accum_loss,
              "train/avg_entropy": avg_entropy,
          }, step=wandb_step[0])
          wandb_step[0] += 1
  ```

  **③ 轮末 eval log 也用同一个 wandb_step 计数器**
  ```python
  wandb.log({
      "eval/accuracy": eval_metrics["accuracy"],
      "eval/format_rate": eval_metrics["format_rate"],
      "eval/rollout_accuracy": rollout_accuracy,
  }, step=wandb_step[0])
  wandb_step[0] += 1
  ```

5. **修复后的效果**

  | 指标 | 改前 | 改后 |
  |------|------|------|
  | `train/sft_loss` | ❌ 不存在 | ✅ 每步一条曲线 |
  | `train/avg_entropy` | ❌ 不存在 | ✅ 能看到模型置信度变化 |
  | `eval/accuracy` | 3 个散点 | ✅ 连续曲线 |
  | wandb x 轴 | `round_idx` (1/2/3) | ✅ 全局 step，连续递增 |

**思考题：为什么参考实现用的是 `step=` 参数，而 GRPO 用的是 `wandb.define_metric`？**

两种做法都可以，区别在于：

| 做法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 显式 `step=` 参数 | 单一时间轴 | 简单直观 | 多个时间轴（rollout step vs train step）时冲突 |
| `define_metric` + metric 值 | 多时间轴 | 不同指标组可以挂在不同 x 轴上 | 需要额外写 define_metric 代码 |

GRPO 需要两个 x 轴（`grpo_step` 给 rollout 指标，`global_step` 给 train/eval 指标），所以用 `define_metric`。EI 只有 SFT 训练一个时间轴，用 `step=` 参数就够了。
