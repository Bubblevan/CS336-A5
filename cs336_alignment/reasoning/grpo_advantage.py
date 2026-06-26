"""
reasoning/grpo_advantage.py

Group Relative Policy Optimization (GRPO) — 组归一化优势计算。

决策链：
    1. 朴素做法：直接把原始奖励当权重做策略梯度
    2. 问题：简单题大家都得高分，难题大家都得低分，梯度被题目难度偏移主导
    3. 所以需要「组内相对好坏」而非「绝对好坏」→ 组归一化优势
    4. 组均值等价于 PPO 中 Critic 网络估计的 V(s)，但不需要额外训练

核心公式：
    A_i = (r_i - μ_group) / (σ_group + ε)       # normalize_by_std=True (标准 GRPO)
    A_i = r_i - μ_group                           # normalize_by_std=False (Dr. GRPO)

形状约定：
    - 输入：rollout_responses / repeated_ground_truths 都是长度为 N*G 的列表
      （N = 问题数, G = 每问题生成数）
    - 输出：advantages 形状 (N*G,)，raw_rewards 形状 (N*G,)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """
    计算 GRPO 组归一化优势。

    流程：
        1. 对每条 (response, ground_truth) 调用 reward_fn，提取 "reward" 分数
        2. reshape 为 (N, G) 矩阵，每行是一个问题的 G 条回答的原始奖励
        3. 每行减组均值 → 得到相对优势（可选的除以组标准差）
        4. 展平回 (N*G,) 对齐 log_probs

    Args:
        reward_fn: 奖励函数，输入 (response, ground_truth) 返回 dict，必须含 "reward" key
        rollout_responses: 模型生成的所有回答，长度 N*G
        repeated_ground_truths: 对应的标准答案，长度 N*G（同一问题的 G 条回答重复同一个 gt）
        group_size: 每个问题生成的回答数 G
        advantage_eps: 防止除零的小常数
        normalize_by_std: True → (r - μ) / σ; False → r - μ（Dr. GRPO）

    Returns:
        advantages:      (N*G,) 组归一化优势张量
        raw_rewards:     (N*G,) 原始奖励张量
        metadata:        含 mean/std/max/min 的统计信息字典，用于 wandb 日志
    """
    # ── 1. 基础校验 ──
    assert len(rollout_responses) == len(repeated_ground_truths), \
        f"rollout_responses ({len(rollout_responses)}) 与 repeated_ground_truths ({len(repeated_ground_truths)}) 数量不一致"
    assert len(rollout_responses) % group_size == 0, \
        f"总样本数 ({len(rollout_responses)}) 必须能被 group_size ({group_size}) 整除"

    # ── 2. 逐条计算原始奖励 ──
    raw_rewards_list: list[float] = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        score_dict = reward_fn(response, truth)
        raw_rewards_list.append(score_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    # shape: (N*G,)

    # ── 3. 按问题分组 ──
    num_questions = raw_rewards.shape[0] // group_size
    grouped_rewards = raw_rewards.view(num_questions, group_size)
    # shape: (N, G)

    # ── 4. 组内归一化 ──
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    # shape: (N, 1) — 广播到 (N, G)

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
        # 使用 Bessel-corrected 标准差（除以 G-1），与 handout snapshot 一致
    else:
        advantages = grouped_rewards - group_means

    # ── 5. 展平 ──
    advantages = advantages.view(-1)
    # shape: (N*G,)

    # ── 6. 元数据（用于 wandb 日志） ──
    metadata: dict[str, Any] = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std(unbiased=False).item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
    }

    return advantages, raw_rewards, metadata
