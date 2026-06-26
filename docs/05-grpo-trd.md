# GRPO：从"正确答案才学"到"所有回答都有用"

## 开篇：GRPO 在做什么？

**GRPO（Group Relative Policy Optimization）** 是 DeepSeekMath 论文中提出的策略梯度算法，核心思想：

> 对同一问题生成 G 条回答 → 计算组内相对优势（比平均好/差多少）→ 用 PPO 的 Clip 机制做策略梯度更新

和 Expert Iteration 的核心区别：

| 特性 | Expert Iteration | GRPO |
|------|-----------------|------|
| 数据利用 | 只保留完全正确的轨迹 | **所有轨迹（包括错的）都参与训练** |
| 梯度来源 | 正确轨迹上的 SFT 交叉熵 | 策略梯度（组归一化优势 × 概率比） |
| 对错误轨迹 | 直接丢弃 | 学"错误轨迹中局部正确的推理" |
| 更新方向 | 模仿正确轨迹 | **增加好轨迹概率，降低差轨迹概率** |
| 复杂度 | 低（纯 SFT） | 高（需要 old_log_probs / clip / advantage） |

EI 的精确率停在 70%，原因之一是它浪费了部分正确的推理路径。GRPO 的组归一化优势让**同一道题的高分回答 push 模型往它靠近，低分回答 push 模型远离它**——这个机制天然就能从"错得没那么离谱的"轨迹中学到信息。

---

## 一、MVP 手把手最小实现

### 1. 第一个愿望：让"错的回答也有用"

我们在 EI 中看到一个问题：Round 1 之后准确率从 54.5% 跳到 69.5%，然后 Round 2/3 就停滞在 70%。分析 rollout 发现，很多错误回答其实是**前半段推理正确、最后算错了**——但 EI 的 filter 直接丢弃了整条轨迹。

所以我们想要的是这样一个训练方式：

```python
for question in dataset:
    responses = model.generate(question, n=8)
    rewards = [grade(r, gt) for r in responses]
    
    # EI: 只保留 reward==1 的
    # GRPO: 所有人都参与，但根据 relative 好坏给不同权重
    
    for r, reward in zip(responses, rewards):
        if reward == 1:
            sft_loss(r)     # 学习好的
        else:
            # reward < 1 的也参与，但权重是负的——让模型远离
            negative_gradient(r, weight=-reward)
```

但直接对错误回答做负梯度有两个问题：数据效率低（每次只能用一个 batch）和策略崩坏风险（一次更新可能让好回答概率骤降）。这就是为什么 GRPO 需要三样东西：**组归一化优势**、**重要性采样比率**、**Clip 截断**。

### 2. 第一个工具：组归一化优势

对于一个问题 q，我们生成了 G 条回答，每条得到原始奖励 rᵢ。GRPO 的核心洞察是：

> 在客观可验证任务中，**同一问题不同回答的奖励差异，才是真正的学习信号**。

PPO 需要训练一个 Critic 网络来估计 $V(s_t)$，需要 4 个模型（Policy + Reference + RM + Critic），显存爆炸。GRPO 用组内均值代替 Critic：

```python
# 第 1 步：收集奖励
raw_rewards = [0, 1, 1, 0, 0, 1, 0, 0]   # 同一问题的 8 条回答
# shape: [8], 每个元素是 0 或 1

# 第 2 步：组归一化（减去组均值，除以组标准差）
group_mean = 0.375    # 8 条里 3 条正确
group_std  = 0.484    # 组内标准差

advantages = (raw_rewards - group_mean) / (group_std + 1e-8)
# advantages = [-0.775, 1.292, 1.292, -0.775, -0.775, 1.292, -0.775, -0.775]
```

这个 advantage 的含义很直观：

- **正 advantage（A > 0）**：这条回答比组内平均水平好 → 我们会 push 模型生成更多类似 token
- **负 advantage（A < 0）**：这条回答比组内平均水平差 → 我们会 push 模型生成更少类似 token

完整实现：

```python
def compute_group_normalized_rewards(
    reward_fn, rollout_responses, repeated_ground_truths,
    group_size, advantage_eps=1e-8, normalize_by_std=True,
):
    # 1. 计算每条回答的原始奖励
    raw_rewards_list = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        score_dict = reward_fn(response, truth)
        raw_rewards_list.append(score_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    # shape: (N * G,)

    # 2. 按问题分组，reshape 为 (N, G)
    num_questions = raw_rewards.shape[0] // group_size
    grouped_rewards = raw_rewards.view(num_questions, group_size)

    # 3. 计算组均值和组标准差
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        # Dr. GRPO: 只减均值，不除标准差
        advantages = grouped_rewards - group_means

    # 4. 展平回 (N * G,)
    advantages = advantages.view(-1)
    return advantages, raw_rewards, metadata
```

**关键问题：为什么减去组均值能消除题目难度偏移？**

简单题大家都得 1 分，难题大家都得 0 分。如果直接用原始奖励作为权重：
- 简单题：8 条都正确，但梯度被 8 份平分，几乎没有学习信号
- 难题：8 条都错误，梯度为 0

减去组均值后，简单题的正确回答优势≈0，错误回答优势≈(|N-1|)/N——模型学到的不是"这道题应该怎么做"，而是"在同类回答中，什么比平均水平好"。

<details>
<summary><strong>还有一个细节：unbiased=False</strong></summary>

`grouped_rewards.std(dim=1, keepdim=True)` 默认使用 `unbiased=False`（总体标准差），除以 G 而非 G-1。因为 G（通常 4-8）很小，无偏修正会放大噪声。这是 GRPO 和统计意义上的标准差的区别——我们不是做统计推断，而是在用一个小的组内样本来做归一化，除以 G 比除以 G-1 更稳定。
</details>

### 3. 第二个工具：三种 Policy Gradient 损失

有了 advantage（组内相对好坏的衡量），我们还需要一个**目标函数**来告诉模型怎么更新参数。

GRPO 有三种损失变体：

**损失类型 A：`no_baseline`（最原始，只用原始奖励）**

```
per_token_objective = log_prob(i,t) * raw_reward(i)
```

不使用任何基线（没有组均值，没有 Critic）。方差极高，因为原始奖励的量级在不同问题之间差异很大。不推荐实际使用，仅作为教学参考。

**损失类型 B：`reinforce_with_baseline`（组内均值做基线）**

```
per_token_objective = log_prob(i,t) * advantage(i)
```

用组内优势作为权重。这是最简化的 GRPO 形式——没有重要性采样，没有 Clip。方差比 no_baseline 小，但仍然存在风险：一次更新用太多 epoch 容易崩。

**损失类型 C：`grpo_clip`（DeepSeekMath 标准）**

这是最终被广泛采用的形式：

```python
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange=0.2):
    # 1. 概率比率（新策略vs旧策略）
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 2. 未截断的 surrogate 目标
    surr1 = ratio * advantages
    
    # 3. 截断后的 surrogate 目标
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = ratio_clipped * advantages
    
    # 4. 取最小值和负号（梯度上升转为梯度下降）
    loss = -torch.min(surr1, surr2)
    
    return loss, {"clip_fraction": ..., "ratio_mean": ...}
```

这个 `loss` 是 per-token 的，形状 `(batch_size, seq_length)`。后续需要聚合（参考下一节）。

**代码中的统一调度入口**：

```python
def compute_policy_gradient_loss(policy_log_probs, loss_type,
                                 raw_rewards=None, advantages=None,
                                 old_log_probs=None, cliprange=None):
    if loss_type == "no_baseline":
        per_token_loss = -policy_log_probs * raw_rewards
    elif loss_type == "reinforce_with_baseline":
        per_token_loss = -policy_log_probs * advantages
    elif loss_type == "grpo_no_clip":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -ratio * advantages
    elif loss_type == "grpo_clip":
        per_token_loss, metadata = compute_grpo_clip_loss(...)
    return per_token_loss, metadata
```

### 4. 第三个工具：长度归一化（Masked Mean vs Masked Normalize）

per-token loss 聚合到 batch 级 loss 有三种方式：

**方式 1：`masked_mean`（原始 GRPO，按回答内部平均）**

每条回答先算自己的 token 平均 loss，再对 batch 内所有回答平均：

```python
def reduce_grpo_objective_mask_mean(per_token_loss, response_mask):
    # per_token_loss: (B, L), response_mask: (B, L)
    # 每条回答的 loss = sum(有效token的loss) / 有效token数
    per_seq_loss = (per_token_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
    # batch 级 loss = 所有回答的平均
    return per_seq_loss.mean()
```

**问题**：长回答中每个 token 的权重被稀释了。一条 1000 token 的错误回答，每个 token 的惩罚只有一条 10 token 错误回答的 1/100。这导致模型倾向于生成更长的回答来稀释惩罚。

**方式 2：`mask_normalize`（Dr. GRPO / Token 平权）**

所有 token 的 loss 加起来，除以一个全局常数 C：

```python
def reduce_grpo_objective_mask_normalize(per_token_loss, response_mask, normalize_constant):
    # 所有有效 token 的 loss 总和，再除以全局常数
    total_loss = (per_token_loss * response_mask).sum()
    return total_loss / normalize_constant
```

C 的选择：
- 对每个 logical batch：`C = response_mask.sum().item()`（batch 内总有效 token 数）
- 对全局训练：`C = max_seq_length`（固定值）

**为什么 Dr. GRPO 用常数 C？**

Understanding R1-Zero-Like Training 明确指出，原始 GRPO 的 `masked_mean` 存在 response length bias：长回答的每个 token 权重低，短回答的每个 token 权重高。这会人为拉长回答，尤其是错误回答——模型通过"堆长度"来碰运气。Dr. GRPO 用常数归一化，让每个 token 平等，自然抑制了无意义的延长。

### 5. 第四个工具：Microbatch Train Step（完整的前向+反向）

有了 loss 函数之后，我们需要一个完整的 train step：

```python
def grpo_microbatch_train_step(
    policy_log_probs,       # (B, L) 当前模型对该 micro-batch 的 log-probs
    response_mask,          # (B, L) 0/1 掩码
    gradient_accumulation_steps,  # 梯度累积步数
    loss_type,              # "grpo_clip" / etc.
    raw_rewards,            # (B, 1) 原始奖励
    advantages,             # (B, 1) 组归一化优势
    old_log_probs,          # (B, L) 旧策略的 log-probs
    cliprange=0.2,          # Clip 阈值 ε
    length_norm_type="mask_normalize",
    normalize_constant=None,
):
    # 1. 计算 per-token 损失
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2. 聚合（长度归一化）
    if length_norm_type == "mask_normalize":
        total_loss = (per_token_loss * response_mask).sum()
        microbatch_loss = total_loss / normalize_constant
        scaled_loss = microbatch_loss  # 已包含梯度累积缩放
    else:  # mask_mean
        per_seq_loss = (per_token_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
        microbatch_loss = per_seq_loss.mean()
        scaled_loss = microbatch_loss / gradient_accumulation_steps

    # 3. 反向传播
    scaled_loss.backward()

    return scaled_loss, metadata
```

### 6. 第五个工具：完整训练主循环

有了以上所有工具后，我们就可以搭建 GRPO 的完整训练循环了。

完整的 GRPO 循环由 4 个阶段组成：

```python
# ── Phase 1: 采样（Rollout）──
policy.eval()
sync_policy_weights(policy, vllm_inst)  # vLLM 权重同步

batch_questions = random.sample(question_pool, rollout_batch_size // group_size)
outputs = vllm_inst.generate(
    [q["prompt"] for q in batch_questions],
    SamplingParams(n=group_size, temperature=1.0, ...)
)

# 展平 vLLM 输出
flat_prompts, flat_responses, repeated_ground_truths = [], [], []
for q_item, out in zip(batch_questions, outputs):
    for candidate in out.outputs:
        flat_prompts.append(q_item["prompt"])
        flat_responses.append(candidate.text)
        repeated_ground_truths.append(q_item["gold"])

# ── Phase 2: 奖励计算 + 组归一化 + 旧策略 log-probs ──
advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
    reward_fn=r1_zero_reward_fn,
    rollout_responses=flat_responses,
    repeated_ground_truths=repeated_ground_truths,
    group_size=group_size,
    normalize_by_std=True,
)

# 分词 + 旧策略 log-probs 预计算
tokenized_data = tokenize_prompt_and_output(flat_prompts, flat_responses, tokenizer)
with torch.no_grad():
    old_log_probs = compute_log_probs_in_microbatches(model, tokenized_data, micro_batch_size)

# ── Phase 3: Inner Loop Training ──
# 在 rollout 数据上做多轮训练
for epoch in range(epochs_per_rollout_batch):
    for logical_indices in iterate_logical_batches(total_samples, train_batch_size, shuffle=True):
        optimizer.zero_grad()
        for micro_step in range(gradient_accumulation_steps):
            micro_indices = logical_indices[micro_step * micro_bs : (micro_step+1) * micro_bs]
            
            # 前向传播
            log_probs_dict = get_response_log_probs(model, input_ids[micro_indices], labels[micro_indices])
            policy_log_probs = log_probs_dict["log_probs"]
            
            # 一个 micro-batch 的 train step（含 backward）
            scaled_loss, meta = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask[micro_indices],
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards[micro_indices].unsqueeze(1),
                advantages=advantages[micro_indices].unsqueeze(1),
                old_log_probs=old_log_probs[micro_indices],
                cliprange=0.2,
            )
        
        # 梯度裁剪 + 优化器更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# ── Phase 4: 评估 + 保存 ──
policy.eval()
sync_policy_weights(policy, vllm_inst)
metrics = log_generations(vllm_inst, eval_params, val_prompts, val_golds, reward_fn)
print(f"Step {step}: eval accuracy = {metrics['eval/accuracy']:.2%}")
model.save_pretrained(f"{output_dir}/grpo_step{step}")
```

### 7. 什么是 On-Policy 和 Off-Policy？

GRPO 的主循环区分 two-policy 和 off-policy 训练：

```
rollout_batch_size = train_batch_size 且 epochs_per_rollout_batch = 1 → On-Policy
rollout_batch_size ≠ train_batch_size 或 epochs_per_rollout_batch > 1 → Off-Policy
```

**On-Policy**：每次 rollout 的数据只用于一个 optimizer step。优点是梯度估计无偏（旧策略 = 新策略），缺点是数据利用率低——每次 rollout 后只更新一步就丢弃。

**Off-Policy**：同一批 rollout 数据用于多次 optimizer step（`epochs_per_rollout_batch > 1`）或组内 sample 多次。优点是数据利用效率高，缺点是存在分布偏移——新策略已经更新了，但 old_log_probs 还是之前保存的。

重要性比率 `ratio = exp(log_prob_new - log_prob_old)` 正是用来修正这个偏移的：当新策略和旧策略不一致时，通过 ratio 来重新加权。而 Clip 机制则确保 ratio 不会偏离 [1-ε, 1+ε] 太远，防止方差爆炸。

值得注意的是 On-Policy 时不能使用 `grpo_clip`，因为 clipping 只在 off-policy 场景下才有必要——on-policy 下 ratio ≈ 1。参考 `cs336-a5-RL/cs336_alignment/grpo.py` 第 134-135 行：

```python
on_policy = epochs_per_rollout_batch == 1 and train_batch_size == rollout_batch_size
assert not (loss_type == "grpo_clip" and on_policy)
```

---

## 二、八股概念基础知识点

### 2.1 GRPO vs PPO vs EI 本质区别

| 对比维度 | PPO | GRPO | Expert Iteration |
|---------|-----|------|-----------------|
| 基线估计 | 需要训练 Critic 网络 $V(s)$ | **组内平均奖励（无需额外训练）** | 无 |
| 所需模型数 | Policy + Reference + RM + Critic = 4 | Policy + Reference = **2** | Policy = 1 |
| 数据利用 | 所有轨迹（含错的）都学 | 所有轨迹都学，**组归一化区分好坏** | 只保留完全正确的 |
| 更新方式 | 策略梯度 + Clip | 策略梯度 + Clip + 组优势 | SFT 交叉熵 |
| 适用场景 | 主观偏好对齐（对话、安全） | 客观可验证任务（数学、代码、推理） | 中等难度推理 |

### 2.2 GRPO 的 4 个关键超参数

**G（Group Size）：每问题生成几条回答**

- G 太小（2-4）：组内统计不稳定，优势估计噪声大
- G 太大（16+）：生成成本高，且组内可能已经有足够的多样性
- 经验值：**G=8**

**ε（Cliprange）：允许的策略更新幅度**

- 太小（0.01）：更新太保守，学习缓慢
- 太大（0.5）：失去约束作用，策略容易崩坏
- 经验值：**ε=0.2**

**normalize_by_std：是否除以组标准差**

- True：标准 GRPO，Z-score 归一化，梯度尺度统一
- False：Dr. GRPO 简化版，防止 G 太小或组奖励高度一致时 std 接近 0
- 经验值：**推荐 True，若发现优势变化剧烈可切换 False**

**length_norm_type：长度归一化方式**

| 类型 | 每个 token 权重 | 回答长度偏置 |
|------|---------------|------------|
| mask_mean | 与回答长度成反比 | 鼓励长回答（稀释 penalty） |
| **mask_normalize** | **所有 token 相等** | **抑制无意义的长回答** |

经验值：**推荐 mask_normalize（Dr. GRPO 建议）**。TRL 文档也已将 dr_grpo 收进 trainer，明确建议用常数而不是序列长度做分母，来消除响应长度偏置。

### 2.3 组归一化优势为什么不需要 Critic

PPO 用 Critic 网络 $V(s)$ 估计"从状态 s 出发的期望回报"，减去状态价值得到优势：$A(s,a) = Q(s,a) - V(s)$。

GRPO 的洞察是：对于同一个问题 q 生成的 G 条回答，组内平均奖励 $\mu = \frac{1}{G}\sum r_i$ 就是该问题初始状态的期望回报的**无偏蒙特卡洛估计**。所以：

$$A_i \approx r_i - \mu = r_i - \frac{1}{G}\sum_j r_j$$

这就等价于 **"用 G 条回答的组内平均代替了 Critic 网络"**。代价是：GRPO 无法在推理过程中间步骤做 advantage 估计（因为直到完整回答结束才有奖励），只能做**序列级**的优势估计。

### 2.4 Clip Fraction 的含义

```python
clipped_mask = (surr2 < surr1).float()  # 1 = 触发了 Clip
clip_fraction = clipped_mask.mean()
```

| clip_fraction | 诊断 |
|---------------|------|
| ≈ 0 | 策略更新过于保守，lr 可能太小 |
| 0.01-0.1 | 健康范围，大部分更新在信任域内 |
| 0.1-0.3 | 边界区间，可观察趋势 |
| > 0.3 | 策略频繁"撞墙"，lr 过大或同一批数据训练太多遍 |

### Interview phrasing

> GRPO 解决了 EI 的"浪费错误轨迹"问题。EI 只保留完全正确的轨迹做 SFT，但很多错误回答的前半段推理是正确的。
>
> GRPO 的核心创新是"组归一化优势"：对同一问题生成 G 条回答，计算组内均值作为基线，每条回答的优势等于它减去组均值再除以组标准差。这样无论题目难易，好的回答都得到正优势，差的回答都得到负优势——梯度贡献是均衡的。
>
> 然后我们用 PPO 的 clipped surrogate objective（加上重要性采样比率 min(ρA, clip(ρ)A)）来做策略梯度更新。Clip 机制确保一次更新不会让策略变化太大。
>
> 长度归一化方面，我倾向于用 Dr. GRPO 的 mask_normalize——用常数 C 做分母，让每个 token 的权重相等，抑制模型无意义地拉长回答来稀释惩罚。

---

## 三、你的项目当前状态：哪些已就绪，哪些需要补

**已就绪的基础设施：**

| 模块 | 文件 | 状态 |
|------|------|------|
| z分词/掩码 | `core/tokenization.py`, `core/masking.py` | ✅ 完整实现 |
| Log-probs 计算 | `core/scoring.py` | ✅ 完整实现 |
| Batching | `core/batching.py` | ✅ 完整实现 |
| 奖励函数 | `reasoning/rewards.py` | ✅ 完整实现 |
| SFT train step | `reasoning/sft.py` | ✅ 完整实现 |
| vLLM 权重同步 | `core/vllm_utils.py` | ✅ 完整实现 |
| 评测生成/GSM8K | `eval/` | ✅ 完整实现 |
| 测试框架 | `tests/` + snapshot | ✅ 使用 handout 适配器 |

**需要实现的核心模块：**

| 模块 | 文件 | 参考实现 |
|------|------|---------|
| 组归一化优势 | `reasoning/grpo_advantage.py` → 实现 `compute_group_normalized_rewards` | `cs336-5/grpo_utils.py:13` |
| PG 损失函数 | `reasoning/grpo_loss.py` → 实现 3 种损失 + 统一调度 | `cs336-5/grpo_utils.py:123-335` |
| Microbatch train step | `reasoning/train_step.py` → 实现完整 forward+backward | `cs336-5/grpo_utils.py:445-503` |
| 提示模板 | `reasoning/prompts.py` → 实现模板加载格式化 | `grpo.ipynb` Cell 9 |
| 主入口 | `run_grpo.py` → 训练循环 + CLI 参数 | `cs336-5/train_grpo.py:88-432` |

**前辈仓库的实现结构：**

两个前辈仓库都完整实现了 GRPO，但组织方式不同：

- **cs336-5**（更完整）：将所有底层函数放在 `grpo_utils.py`，`train_grpo.py` 负责训练循环 + CLI。有完整的消融实验脚本（`run_grpo_baseline.sh`、`run_grpo_clip.sh` 等）。
- **cs336-a5-RL**（更紧凑）：将所有功能（grpo loss + 训练循环）放在 `grpo.py`，基础设施放在 `helpers.py`。训练入口直接用 `python cs336_alignment/grpo.py`。

你的项目结构和 cs336-5 更接近，可以重点参考 cs336-5 的代码组织方式。

---

## 四、排障过程实践

### 现象 1：pytest 测试失败

**表现**：
```bash
cd /root/gpufree-data/cs336
pytest tests/test_grpo.py -v
```

输出大量 `NotImplementedError`，因为 `adapters.py` 还没有实现 GRPO 相关函数。

**根因**：`tests/adapters.py` 的 `run_compute_group_normalized_rewards`、`run_grpo_train_step` 等方法都是 `raise NotImplementedError`。

**解决**：实现 `reasoning/grpo_advantage.py`、`reasoning/grpo_loss.py`、`reasoning/train_step.py` 后，修改 `tests/adapters.py` 导入真正的实现、替换桩函数。

但在实际开发中，**建议先跑一遍已有的核心测试**确认基础设施 intact：

```bash
cd /root/gpufree-data/cs336
pytest tests/ -v --ignore=tests/test_grpo.py
```

### 现象 2：test_grpo.py 中有三个维度不同的 GRPO 测试

`tests/test_grpo.py` 中有 3 个组的 snapshot 测试：

```python
# 1. On-Policy 标准测试
test_grpo_train_step_standard_on_policy

# 2. 多种变体（On-Policy with different loss types）
test_grpo_train_step_variants_on_policy
# 参数化：grpo_constant, maxrl, dr_grpo, rft

# 3. Off-Policy 测试
test_grpo_train_step_off_policy
# 参数化：grpo, gspo, noclip
```

这对应了算法的三种训练模式：
- **On-Policy**：rollout_batch_size == train_batch_size，epochs_per_rollout_batch == 1
- **On-Policy with variants**：不同 loss normalization（constant, maxrl, dr_grpo, rft）
- **Off-Policy**：不同 importance reweighting（grpo, gspo, noclip）

### 现象 3：rollout 阶段 vLLM 权重同步

**表现**：训练几轮后验证 accuracy 没有提升，甚至掉回初始值。

**根因**：vLLM 的推理权重没有随 HF 训练权重同步。GRPO 每轮 rollout 需要先用当前最新的 policy 权重重新生成回答。

**正确做法**：

```python
# Phase 1: Rollout 前必须 sync
policy.eval()
sync_policy_weights(policy, vllm_inst)

outputs = vllm_inst.generate(prompts, rollout_sampling_params)
# ...

# Phase 3: 训练
policy.train()
# ... inner loop ...

# Phase 4: 评估前也必须 sync
policy.eval()
sync_policy_weights(policy, vllm_inst)
metrics = evaluate(vllm_inst, val_prompts, ...)
```

参考 `cs336-5/cs336_alignment/train_grpo.py`：
- L153: `load_policy_into_vllm_instance(policy, vllm_inst)` — 初始评估前
- L189: `load_policy_into_vllm_instance(policy, vllm_inst)` — 每轮 rollout 前
- L409: `load_policy_into_vllm_instance(policy, vllm_inst)` — 每轮评估前

如果项目使用 `vllm_utils.py` 的 `sync_policy_weights`（NCCL 权重同步），需要确保：
1. 先调用 `vllm_server.init_weight_sync(device)` 初始化权重同步组
2. 之后每次调用 `vllm_server.sync_policy_weights(policy)`

如果使用 `cs336-5` 的 `load_policy_into_vllm_instance`（通过模型拷贝），则在单卡模式下可能需要更多优化来管理显存。

### 现象 4：rollout 结果全部为正确/全部为错误

**表现**：某一轮 rollout 的 `mean_reward` 接近 1 或 0。

**根因**：
- 全部正确：`sampling_temperature` 太低（接近 0），导致生成缺乏多样性，尤其当模型已经很强时
- 全部错误：`sampling_temperature` 太高（> 1.5），或者模型在 RL 训练中发生 collapse

**解决**：
```python
# Exploration 阶段：temperature 适中
rollout_params = SamplingParams(
    n=group_size,
    temperature=0.6,    # 不要用 0.0
    min_tokens=4,
    max_tokens=1024,
)

# Eval 阶段：greedy
eval_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
)
```

如果发现 rollout 多样性不足但模型准确率已经较高（>70%），可以尝试 `top_p=0.95` 结合 temperature 来增加探索。

### 现象 5：Inner Loop 训练 loss 震荡或爆炸

**表现**：train loss 在几步内从 0.1 跳到 100+，然后模型输出乱码。

**根因**：学习率过大或 epochs_per_rollout_batch 太多。GRPO 对学习率非常敏感。

**经验范围**（参考 `cs336-5/cs336_alignment/run_grpo_baseline.sh` 和 `cs336-a5-RL/cs336_alignment/grpo.py` 的 sweep）：

| 模型大小 | 学习率范围 |
|----------|-----------|
| 1.5B（Qwen2.5-Math） | LR: 3e-6 ~ 5e-5（cs336-5 最佳: 3e-5） |
| 7B+ | LR: 1e-6 ~ 1e-5 |

**稳定策略**：
1. 开启梯度裁剪 `clip_grad_norm=1.0`
2. `epochs_per_rollout_batch` 从 1 开始（严格 on-policy），确认稳定后再增加
3. 当 clip_fraction > 0.3 时，立即降低学习率或减少 epoch
4. 如果使用 `grpo_clip`，确保 `rollout_batch_size != train_batch_size`（否则是 on-policy），参考 `cs336-a5-RL` 的断言

### 现象 6：显存不足 (OOM) — CUDA 0/1 傻傻分不清

**表现 1（你的 case）**：报错说 GPU 0 CUDA OOM，但降低 `--vllm_gpu_util` 从 0.95 到 0.9 就解决了。
```
RuntimeError: CUDA out of memory. Tried to allocate ... on GPU 0
```
但调低的明明是 GPU 1 的 vLLM 占用——为什么 GPU 0 的 OOM 靠调 GPU 1 的参数修好了？

**根因**：**NCCL 权重同步需要双卡同时分配 buffer**。

流程是这样的：

```
Phase 1: Rollout 前 sync_policy_weights()
  ├─ GPU 0 (训练): 从 policy model 收集所有权重 → 分配 ~3GB send buffer
  └─ GPU 1 (vLLM): 分配 ~3GB receive buffer ──→ ❌ gpu_memory_util=0.95 下只剩 2.3GB
  
NCCL AllReduce 失败 → 错误传到 GPU 0 → 表现成 GPU 0 OOM
```

GPU 1 被 vLLM 占满了（0.95 × 46GB = 43.7GB），只剩 ~2.3GB，不够 NCCL 分配 ~3GB 的 receive buffer。NCCL 是同步通信，任一卡失败都会导致所有卡报错——所以错误会表现为 GPU 0 OOM。

**修复**：`--vllm_gpu_util 0.95` → `0.9`，vLLM 少占 2.3GB，给 NCCL sync 留出 ~4.6GB 空余。

```
0.95 × 46GB = 43.7GB → 剩余 2.3GB  ❌ NCCL 需要 ~3GB
0.90 × 46GB = 41.4GB → 剩余 4.6GB  ✅ NCCL 够用
```

**经验规则**：双卡 GRPO 的 `vllm_gpu_util` 经验值在 **0.85-0.90** 之间。太高（>0.93）会导致 NCCL sync OOM，太低（<0.8）浪费显存。

---

**表现 2（常规 OOM）**：Phase 3（inner loop 训练）报 CUDA OOM。

**根因**：micro_batch_size 太大，或序列太长（max_tokens 太大）。

**解决**：
- 减少 `micro_batch_size`（`train_batch_size // grad_accum_steps`，逐步减半）
- 减少 `sampling_max_tokens`（1024 → 512，但可能截断有效推理）
- 开启 `gradient_checkpointing`（已默认开启）

**OOM 通用检查点**：
```bash
# 监控显存
nvidia-smi

# 减少 micro_batch_size（逐步减半）：
# train_batch_size=32, grad_accum_steps=8 → micro_batch_size=4 → 2

# 减少 max_tokens：
# sampling_max_tokens=1024 → 512 （但可能截断有效推理）

# 开启 gradient_checkpointing：
# 已默认开启（policy.gradient_checkpointing_enable()）
```

### 现象 7：验证 accuracy 不提升

**表现**：GRPO 跑了几十步，reward 在涨但 eval accuracy 不变或下降。

**根因**：reward hacking。模型学会"让自己在组内相对更好"而不是"做对题目"。常见 case：
- 模型生成更长的回答（就算错了也看起来更"认真"），组内平均被拉低，导致长错误回答反而没得到足够惩罚
- 模型在格式上做文章（更有完整的 `<think>` 标签），让格式奖励（format_reward）提升而答案奖励不变

**诊断**：
```python
# 在 log_generations 中分开记录
metrics = {
    "eval/accuracy": accuracy,        # reward == 1 的比例
    "eval/avg_format_reward": ...,    # 格式得分
    "eval/avg_answer_reward": ...,    # 答案正确率  
    "eval/avg_response_length": ...,  # 平均生成长度
}
```

**解决**：
1. 检查 `avg_answer_reward` 是否稳步提升——这是真正的推理能力提升
2. 检查 `avg_response_length`——如果大幅增加但 accuracy 没变，说明模型在"堆长度"
3. 切换到 `mask_normalize`（Dr. GRPO 的长度归一化），消除长度偏置
4. 在 reward function 中对过长回答加 penalty（但会引入额外调试成本）

---

## 从 EI 到 GRPO：70% → ?

```
SFT (22.5% → 54.5%) → EI (54.5% → 70.0%) → GRPO (70.0% → ?)
```

EI 的局限：只保留完全正确的轨迹，浪费了部分正确的推理。

GRPO 的优势：**所有轨迹参与训练**，通过组归一化优势区分好坏，让模型学会"在同一道题中哪些 token 导致了更好或更差的结果"。

预期 GRPO 在 MATH 上的提升空间：
- 第一个 5-10 步：可能快速跳到 75-80%（模型适应策略梯度信号）
- 后续收敛：取决于模型容量和 rollout 多样性，可能会到 80-85%

---

## 面试表达

```
GRPO 不是 PPO 的魔改版，而是为客观可验证任务做的减法。

PPO 需要 4 个模型（Policy + Reference + RM + Critic），其中 Critic 的职能
是估计状态价值 V(s)。但在推理任务中，对同一问题采样多条回答，
它们的平均奖励就是对初始状态期望回报的无偏估计——所以 Critic 完全多余。

GRPO 的核心决策链是：
1. 我不想训练 Critic → 用组内平均奖励代替基线
2. 我想用所有轨迹（包括错的）→ 用组归一化优势区分好坏  
3. 我想复用旧数据（off-policy）→ 引入重要性采样比率
4. 我怕方差爆炸 → 用 PPO 的 Clip 限制更新幅度
5. 我担心长度偏置 → 用常量归一化代替序列平均

每一步都有一个明确的失败模式驱动：显存不够逼出组均值代替 Critic，
数据效率低逼出重要性采样，策略崩坏逼出 Clip，偏置逼出长度归一化。
不是先选了 GRPO 这个名词，而是从问题长出来的。
```

---

## 四、实战结果：EI 70.0% → GRPO 82.5%

> **实验记录**: [wandb 面板](https://wandb.ai/bbw486502970-zhejiang-university/cs336-grpo/runs/8s1gxxj5)

### 实验配置

| 参数 | 值 |
|------|-----|
| 基座 | Qwen2.5-Math-1.5B-EI-round3（MATH 70%） |
| GPU | 2×L40（cuda:0 HF 训练，cuda:1 vLLM 生成，NCCL 权重同步） |
| group_size | 8 |
| rollout_batch_size | 256（每步抽 32 题 × 8 条） |
| train_batch_size | 32，grad_accum_steps=8 |
| lr | 3e-5 |
| loss_type | grpo_clip（ε=0.2） |
| length_norm_type | mask_normalize |
| 训练步数 | 60 步（手动终止，原计划 100） |
| 耗时 | ~62min |

### 逐步结果

| GRPO Step | Eval Acc | Rollout Mean Reward | Unique Ratio | Train Loss | Clip Frac | 备注 |
|-----------|----------|-------------------|--------------|------------|-----------|------|
| 0 | 70.5% | — | — | — | — | EI round3 基线 |
| 5 | 70.0% | 0.61 | 0.82 | -0.15 | 0.25 | 起步，loss 负值正常（优势正token多） |
| 10 | 73.5% | 0.55 | 0.92 | 0.22 | 0.12 | 稳步上升 |
| 15 | **76.5%** | 0.71 | 0.87 | 0.06 | 0.16 | GRPO 首次有效提升 |
| 20 | 79.5% | 0.76 | 0.94 | 0.28 | 0.12 | +9pp |
| 25 | **81.5%** 🏆 | 0.61 | 0.99 | 0.19 | 0.09 | 局部高点 |
| 30 | 79.5% | 0.81 | 0.99 | 0.08 | 0.04 | 震荡 |
| 35 | 80.5% | 0.79 | 0.98 | 0.41 | 0.11 | |
| 40 | **82.5%** 🏆🏆 | 0.77 | 0.97 | 2.71 | 0.05 | **全局最高点** |
| 45 | 80.5% | 0.67 | 0.99 | 24982 | 0.07 | **梯度爆炸** |
| 50 | 80.5% | 0.73 | 0.98 | 14870 | 0.12 | loss 高位震荡 |
| 55 | 74.5% | 0.58 | 0.99 | 45.4 | 0.20 | 开始崩溃 |
| 60 | **59.0%** 💥 | 0.50 | 0.99 | 385 | 0.56 | clip 疯狂触发 |

### 可视化

```
   85% │                                🏆
       │                                │
   80% │              🏆──●──●           ●
       │             ╱           ╲     ╱
   75% │      ●────●             ●──●
       │     ╱
   70% │ ●──●                                   ●
       │                                          ╲
   65% │                                           ╲
       │                                            ╲
   60% │                                             ●💥
       │
       └───────────────────────────────────────────────▶ GRPO Step
       0   5   10  15  20  25  30  35  40  45  50  55  60
```

### 关键发现

**1. GRPO 有效：EI 70% → GRPO 82.5%，+12.5pp**

40 步内从 70.5% 拉到 82.5%，说明 GRPO 的组归一化优势 + clip 机制在 MATH 推理任务上确实能突破 EI 的上限。EI 只学完全正确的轨迹（丢弃了部分正确的），GRPO 通过组内相对好坏区分度，让"错得没那么离谱"的推理路径也贡献了学习信号。

**2. Step 36 单步 loss=4.5M，梯度爆炸导致崩盘**

loss 从 step 35 的 0.41 跳到 step 36 的 4,588,314（千万倍增幅）。跟踪 cause：

```text
ratio = exp(log_prob_new - log_prob_old)

新策略在某几个 token 上概率从 0.001 → 0.5
→ ratio = exp(ln(0.5) - ln(0.001)) = exp(6.2) ≈ 500
→ per_token_loss = -min(ratio·A, clip(ratio)·A)
→ ratio·A = 500 × 0.707 = 353（未 clip 的 surrogate 项）
→ clip(ratio)·A = 1.2 × 0.707 = 0.85（clip 后的 surrogate 项）
→ min 选 0.85（clip 生效了，但单步参数变化已经太大）

clip 虽然截断了这一项，但 4.5M 的 loss 意味着其他 token 的
ratio 比 500 还大——clip 在 loss 层面保护了梯度幅度，
但参数更新已经偏离旧策略太远，后面再也回不来。
```

**3. 多样性修复生效**

`rollout/unique_ratio` 全程 0.82~1.0，同题 8 条回答中多数不同，组内 advantage 估计有效。seed 修复生效，GRPO 不是白跑的。

**4. clip_fraction 是崩溃的预警指标**

| clip_fraction | 诊断 |
|---------------|------|
| 0.04~0.15 | ✅ 健康 |
| 0.15~0.25 | ⚠️ 边界 |
| **>0.25** | 🔴 **策略即将崩溃**（step 36 前 clip=0.25，此后 clip 持续 >0.5） |
| >0.5 | 💥 策略已崩 |

### 崩溃根因分析：为什么只有 clip 不够？

两个前辈仓库（[cs336-5](https://github.com/cs336-spring2025/cs336-5)、[cs336-a5-RL](https://github.com/cs336-spring2025/cs336-a5-RL)）的 GRPO 实现**也都没有 KL 散度约束**，和本项目一样只靠 `grpo_clip` 的 clip 机制。但为什么他们没崩？

**可能性 1：学习率差异**

| 实现 | 模型大小 | LR | 是否崩 |
|------|---------|-----|--------|
| cs336-5 | 1.5B | 3e-5 | ✅ 可能是最佳区间 |
| cs336-a5-RL | 1.5B | **5e-5** | 可能也崩（未公开日志） |
| 本项目 | 1.5B | 3e-5 | ❌ 崩了 |

本次实验 lr=3e-5 和 cs336-5 的最佳 LR 一致，但还是崩了。所以单纯低 LR 不够。

**可能性 2：rollout_batch_size × epochs_per_rollout_batch**

cs336-5 的 `rollout_batch_size=256` 且 `epochs_per_rollout_batch=1`（严格 on-policy），同配置。不是这个。

**可能性 3：rollout 多样性导致更新幅度差异**

EI 之后模型在 MATH 上已经 70%，正确回答比例高。当 group_size=8 时，可能 6/8 条正确、2/8 条错误，组均值和组标准差使得"正确 - 平均" ≈ 0，"错误 - 平均" ≈ 0。但偶尔出现一条**所有 token 都获得大幅正向梯度**的情况→策略某几个 token 概率骤升→ratio 爆炸。

更根本地说：**clip 只是在 loss 计算时把 `ratio` 限制在 [0.8, 1.2] 区间内，但它不阻止多步累积的偏移**。每一步都更新一点点，多步之后策略和旧策略已经是完全不同的人。KL 散度约束或 adaptive KL 可以在参数空间中直接惩罚偏离程度。

**可能性 4：DeepSeekMath 原论文的 GRPO 是否包含 KL？**

DeepSeekMath 原论文的 GRPO 目标函数：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \min(\rho_t A_i, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_i) \right] \mathbf{- \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})}$$

是的，**原论文包含 KL 散度项**，通过 reference model 计算新旧策略的 KL 散度作为正则项。而课堂 handout 的 GRPO 实现简化了这一步——去掉了 reference model 和 KL 惩罚，只保留 clip。这是本次训练崩溃的根本原因。

### 已实现的稳定性修复

第一次训练崩溃后，以下两个工程修复已实装到 `run_grpo.py`：

#### 1. KL 散度约束（`--kl_coef`）

```
新增 CLI 参数 --kl_coef，默认 0.0（禁用）

当 kl_coef > 0 时：
  1. 额外加载一个 reference model（与 policy 同架构，冻住不更新梯度）
  2. 每个 micro-batch 前向时，同时用 ref_model 计算 ref_log_probs
  3. 计算逐 token 的近似 KL：ρ - log(ρ) - 1，其中 ρ = exp(ref_lp - policy_lp)
  4. 将 β * KL 加到总 loss 中，与 PG loss 共享梯度流

KL 散度的作用是：当策略试图偏离初始模型太远时，
KL 项会产生一个"往回拉"的梯度，从根本上抑制策略漂移。
和 clip 不同（clip 只截断单步更新幅度），KL 是参数空间中的软约束。
```

参考模型加载占用 ~3GB 显存（bf16 1.5B），双卡场景下有足够余量。

#### 2. Early Stopping（`--early_stopping_patience`）

```
新增 CLI 参数 --early_stopping_patience，默认 3，0=禁用

训练过程中：
  1. 每次 eval 结束后记录当前 accuracy
  2. 若 accuracy > best，更新 best、重置计数器
  3. 若 accuracy ≤ best，计数器 +1
  4. 当计数器 ≥ patience 时，打印日志并跳出 GRPO 循环
```

#### 用法示例

```bash
uv run python -m cs336_alignment.run_grpo \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B-EI-round3 \
    --train_data /root/gpufree-share/data/MATH/train.jsonl \
    --val_data /root/gpufree-share/data/MATH/validation.jsonl \
    --prompt_path cs336_alignment/prompts/r1_zero.prompt \
    --device cuda:0 --engine vllm --vllm_device cuda:1 \
    --group_size 8 --rollout_batch_size 256 \
    --train_batch_size 32 --grad_accum_steps 8 \
    --lr 3e-5 --n_grpo_steps 100 \
    --loss_type grpo_clip --length_norm_type mask_normalize \
    --kl_coef 0.04 \
    --early_stopping_patience 3 \
    --eval_every_steps 5 --save_every_steps 25 \
    --eval_limit 200 \
    --wandb_project cs336-grpo \
    --output_dir outputs/grpo_v2
```
