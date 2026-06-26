"""
reasoning/train_step.py

grpo_microbatch_train_step — GRPO 微批次训练步。

功能：
    接收当前策略的 token log-probs 和已计算的奖励/优势，
    计算策略梯度损失 → 聚合 → 反向传播。

决策链：
    1. 朴素做法：在 loss 函数外手动 backward(loss)
    2. 问题：梯度累积需要按步数缩放 loss，且需要统一聚合方式
    3. 所以：把 "计算 loss + 聚合 + 缩放 + backward" 封成一个步骤
    4. 聚合方式又引出了 mask_mean vs mask_normalize 的选择

和 `reasoning/grpo_loss.py` 的关系：
    grpo_loss：纯函数，输入 log-probs 输出 per-token loss，不涉及 backward
    train_step：在此基础上 + 聚合 + 缩放 + backward，包含梯度计算图操作
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from cs336_alignment.core.masking import masked_mean
from cs336_alignment.reasoning.grpo_loss import compute_policy_gradient_loss


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    length_norm_type: str = "mask_mean",
    normalize_constant: float | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """
    单微批次 GRPO 训练步。

    流程：
        1. 通过 compute_policy_gradient_loss 计算 per-token 损失
        2. 按 length_norm_type 聚合为标量
        3. 按梯度累积步数缩放
        4. 调用 backward()

    Args:
        policy_log_probs:        (B, L) 当前策略的对数概率
        response_mask:           (B, L) 0/1 掩码，1 表示 response 有效 token
        gradient_accumulation_steps:  梯度累积步数，loss 需除以此值
        loss_type:               "no_baseline" / "reinforce_with_baseline" / "grpo_clip"
        raw_rewards:             (B, 1) 原始奖励，no_baseline 时需要
        advantages:              (B, 1) 组归一化优势，reinforce_with_baseline / grpo_clip 时需要
        old_log_probs:           (B, L) 旧策略对数概率，grpo_clip 时需要
        cliprange:               ε，grpo_clip 时需要
        length_norm_type:        "mask_mean" / "mask_normalize"
        normalize_constant:      当 length_norm_type="mask_normalize" 时的分母常数

    Returns:
        scaled_loss:  已缩放的可调用 backward() 的标量损失
        metadata:     含原始损失和 loss_metadata 的字典
    """
    # ── 1. 计算 per-token 策略梯度损失 ──
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    # shape: (B, L)

    # ── 2. 聚合（长度归一化） ──
    if length_norm_type == "mask_normalize":
        # Token 平权：所有有效 token 求和，除以全局常数
        assert normalize_constant is not None, "mask_normalize 需要 normalize_constant"
        total_loss = (per_token_loss * response_mask).sum()
        microbatch_loss = total_loss / normalize_constant
        scaled_loss = microbatch_loss  # 已包含全局分母，不再除 grad_accum
    else:
        # Sequence 平权（默认）：每条回答内平均，再跨回答平均
        per_seq_loss = masked_mean(per_token_loss, response_mask, dim=1)
        # shape: (B,)
        microbatch_loss = per_seq_loss.mean()
        scaled_loss = microbatch_loss / gradient_accumulation_steps

    # ── 3. 反向传播 ──
    scaled_loss.backward()

    # ── 4. 返回 ──
    metadata: dict[str, Any] = {
        "loss": microbatch_loss.detach(),
    }
    metadata.update(loss_metadata)

    return scaled_loss, metadata
