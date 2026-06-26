"""
reasoning/grpo_loss.py

GRPO 策略梯度损失函数。

三种损失模式（从简到繁）：
    1. no_baseline（REINFORCE）：直接 log_prob × 原始奖励，无基线
    2. reinforce_with_baseline：log_prob × 组归一化优势（带基线）
    3. grpo_clip（DeepSeekMath 标准）：重要性采样比率 + PPO 式 Clip

决策链：
    1. 朴素 REINFORCE：log_prob × reward，方差极大
    2. 减基线降方差 → reinforce_with_baseline（用组均值代替 Critic）
    3. 需要 Off-Policy 复用数据 → 重要性采样比率 ratio = exp(new - old)
    4. 比率过大导致方差爆炸 → 加入 Clip 截断 [1-ε, 1+ε]

形状约定：
    - policy_log_probs / old_log_probs:  (B, L) 逐 token 对数概率
    - raw_rewards / advantages:          (B, 1) 每条回答标量，广播到 (B, L)
    - 返回: per_token_loss               (B, L) 逐 token 损失
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from cs336_alignment.core.masking import masked_mean


# ═══════════════════════════════════════════════════════════════════
# 1. compute_naive_policy_gradient_loss（无重要性采样）
# ═══════════════════════════════════════════════════════════════════


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """
    最基础的策略梯度损失：- Σ log_prob(a|s) × R。

    对应 importance_reweighting_method="none"（on-policy）。
    既不含重要性采样比率，也不含 Clip 截断。

    Args:
        raw_rewards_or_advantages: (B, 1) 每条回答的原始奖励或优势
        policy_log_probs:          (B, L) 当前策略的对数概率

    Returns:
        per_token_loss: (B, L) 逐 token 策略梯度损失
    """
    # raw_rewards_or_advantages: (B,1) → (B,1) → 广播到 (B,L)
    # -log_prob * reward：梯度上升 → 取负号转为梯度下降
    return -policy_log_probs * raw_rewards_or_advantages


# ═══════════════════════════════════════════════════════════════════
# 2. compute_grpo_clip_loss（重要性采样 + Clip）
# ═══════════════════════════════════════════════════════════════════


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    PPO/GRPO 风格的 Token-Level Clip 损失。

    公式（DeepSeekMath 论文 Eq.28 的 clipped surrogate）：
        ratio = exp(log_prob_new - log_prob_old)
        surr1 = ratio * A
        surr2 = clip(ratio, 1-ε, 1+ε) * A
        loss  = -min(surr1, surr2)

    Clip 物理意义：
        - A > 0（好回答）：限制概率增加，防止一次更新暴涨
        - A < 0（差回答）：限制概率减少，防止永久扼杀潜在合理路径

    Args:
        advantages:      (B, 1) 组归一化优势
        policy_log_probs: (B, L) 当前策略对数概率
        old_log_probs:    (B, L) 旧策略对数概率（rollout 时缓存）
        cliprange:        ε，通常 0.2

    Returns:
        per_token_loss: (B, L) 逐 token 损失
        metadata:       含 clip_fraction / ratio_mean / ratio_max / ratio_min
    """
    # ── 1. 概率比率 ──
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    # shape: (B, L)

    # ── 2. 两个 surrogate 项 ──
    # advantages: (B, 1) → (B, L) 广播
    surr1 = ratio * advantages

    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = ratio_clipped * advantages

    # ── 3. 悲观下界 → 取负号 → 梯度下降 ──
    per_token_loss = -torch.min(surr1, surr2)
    # shape: (B, L)

    # ── 4. 元数据 ──
    with torch.no_grad():
        # clip_fraction：多少比例的 token 触发了截断
        clipped_mask = (surr2 < surr1).float()  # 1 = 选了截断后的值
        clip_fraction = clipped_mask.mean()

        metadata: dict[str, Tensor] = {
            "clip_fraction": clip_fraction,
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
            "kl_approx": (ratio * log_ratio - ratio + 1.0).mean(),
        }

    return per_token_loss, metadata


# ═══════════════════════════════════════════════════════════════════
# 3. compute_policy_gradient_loss（统一调度入口）
# ═══════════════════════════════════════════════════════════════════


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """
    统一调度入口：根据 loss_type 分发到对应的损失函数。

    对应 handout 的三种 loss_type 与四种 importance_reweighting_method 的映射：
        no_baseline          → 原始 REINFORCE（importance_reweighting_method="none"）
        reinforce_with_baseline → 带基线 REINFORCE（组优势，无重要性采样）
        grpo_clip            → GRPO-Clip（重要性采样 + token-level clip）

    Args:
        policy_log_probs: (B, L) 当前策略对数概率
        loss_type:        "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
        raw_rewards:      (B, 1) 原始奖励，仅 no_baseline 使用
        advantages:       (B, 1) 组归一化优势，reinforce_with_baseline / grpo_clip 使用
        old_log_probs:    (B, L) 旧策略对数概率，仅 grpo_clip 使用
        cliprange:        截断阈值 ε，仅 grpo_clip 使用

    Returns:
        per_token_loss: (B, L) 逐 token 损失
        metadata:       含损失统计信息的字典
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline 需要 raw_rewards"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline 需要 advantages"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    elif loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip 需要 advantages"
        assert old_log_probs is not None, "grpo_clip 需要 old_log_probs"
        assert cliprange is not None, "grpo_clip 需要 cliprange"
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    else:
        raise ValueError(f"未知 loss_type: {loss_type}")
