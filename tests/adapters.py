from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    from cs336_alignment.core.tokenization import tokenize_prompt_and_output
    return tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    from cs336_alignment.core.scoring import get_response_log_probs
    return get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    from cs336_alignment.reasoning.rewards import r1_zero_reward_fn
    raw_rewards = []
    format_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, gt) if reward_fn is not None else r1_zero_reward_fn(response, gt)
        raw_rewards.append(scores.get("reward", 0.0))
        format_rewards.append(scores.get("format_reward", 0.0))
    raw_rewards_t = torch.tensor(raw_rewards)
    metadata = {
        "mean_reward": float(raw_rewards_t.mean()),
        "mean_format_reward": float(torch.tensor(format_rewards).mean()),
    }
    return raw_rewards_t, metadata


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    total = raw_rewards.shape[0]
    assert total % group_size == 0, \
        f"Total samples ({total}) must be divisible by group_size ({group_size})"
    num_questions = total // group_size

    grouped = raw_rewards.view(num_questions, group_size)  # (N, G)

    # ── 1. Baseline subtraction ──
    if baseline == "mean":
        group_means = grouped.mean(dim=1, keepdim=True)
        centered = grouped - group_means
    else:  # "none"
        centered = grouped

    # ── 2. Normalization ──
    if advantage_normalizer == "std":
        group_stds = grouped.std(dim=1, keepdim=True)
        advantages = centered / (group_stds + advantage_eps)
    elif advantage_normalizer == "mean":
        group_means = grouped.mean(dim=1, keepdim=True)
        advantages = centered / (group_means.abs() + advantage_eps)
    else:  # "none"
        advantages = centered

    advantages = advantages.view(-1)  # (N*G,)

    metadata: dict[str, float] = {
        "mean_advantage": float(advantages.mean().item()),
        "std_advantage": float(advantages.std(unbiased=False).item()),
        "mean_reward": float(raw_rewards.mean().item()),
        "std_reward": float(raw_rewards.std(unbiased=False).item()),
    }

    return advantages, metadata


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    B, L = policy_log_probs.shape
    _ = L

    # Expand reward/advantage from (B,) or (B,1) to (B,1) for broadcasting
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)  # (B, 1)

    # ── Method "none": on-policy, no importance sampling ──
    if importance_reweighting_method == "none":
        per_token_loss = -policy_log_probs * raw_rewards_or_advantages
        # (B, 1) broadcast to (B, L)
        return per_token_loss, {}

    # ── Methods requiring old_log_probs ──
    assert old_log_probs is not None, f"{importance_reweighting_method} 需要 old_log_probs"

    log_ratio = policy_log_probs - old_log_probs  # (B, L)

    if importance_reweighting_method == "noclip":
        # 重要性采样权重但不截断
        ratio = torch.exp(log_ratio)
        per_token_loss = -ratio * raw_rewards_or_advantages
        return per_token_loss, {}

    elif importance_reweighting_method == "grpo":
        # Token-level clipping (标准 GRPO)
        assert cliprange is not None, "grpo 需要 cliprange"
        ratio = torch.exp(log_ratio)
        ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        surr1 = ratio * raw_rewards_or_advantages
        surr2 = ratio_clipped * raw_rewards_or_advantages
        per_token_loss = -torch.min(surr1, surr2)

        with torch.no_grad():
            clipped_mask = (surr2 < surr1).float()
            metadata: dict[str, torch.Tensor] = {
                "clip_fraction": clipped_mask.mean(),
                "ratio_mean": ratio.mean(),
            }
        return per_token_loss, metadata

    elif importance_reweighting_method == "gspo":
        # Sequence-level clipping (GSPO)
        assert cliprange is not None, "gspo 需要 cliprange"
        assert response_mask is not None, "gspo 需要 response_mask"

        # 序列级 log ratio = 对 response token 求平均
        seq_log_ratio = (log_ratio * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
        # (B,)
        seq_ratio = torch.exp(seq_log_ratio)  # (B,)
        seq_ratio_clipped = torch.clamp(seq_ratio, 1.0 - cliprange, 1.0 + cliprange)

        # Expand back to (B, 1) for broadcasting
        surr1 = seq_ratio.unsqueeze(1) * raw_rewards_or_advantages  # (B, 1)
        surr2 = seq_ratio_clipped.unsqueeze(1) * raw_rewards_or_advantages
        # All tokens in the same sequence get the same loss, expand to (B, L)
        per_token_loss = -torch.min(surr1, surr2).expand(-1, L)  # (B, L)

        with torch.no_grad():
            seq_clipped = (surr2 < surr1).float()
            metadata = {
                "clip_fraction": seq_clipped.mean(),
                "ratio_mean": seq_ratio.mean(),
            }
        return per_token_loss, metadata

    else:
        raise ValueError(f"未知 importance_reweighting_method: {importance_reweighting_method}")


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if loss_normalization == "sequence":
        # 每条回答内平均，再跨回答平均
        per_seq_loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return per_seq_loss.mean()
    elif loss_normalization == "constant":
        assert normalization_constant is not None, "constant 模式需要 normalization_constant"
        # 所有有效 token 求和，除以常数
        total_loss = (per_token_policy_gradient_loss * mask).sum()
        return total_loss / normalization_constant
    else:
        raise ValueError(f"未知 loss_normalization: {loss_normalization}")


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    from cs336_alignment.core.tokenization import tokenize_prompt_and_output
    from cs336_alignment.core.scoring import get_response_log_probs

    # ── 1. Tokenize ──
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]       # (B, L)
    labels = tokenized["labels"]             # (B, L)
    response_mask = tokenized["response_mask"]  # (B, L)
    B, L = input_ids.shape

    # ── 2. Compute raw rewards ──
    raw_rewards_list = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, gt)
        raw_rewards_list.append(scores.get("reward", 0.0))
    raw_rewards_t = torch.tensor(raw_rewards_list, dtype=torch.float32)  # (B,)

    # ── 3. Compute advantages (only if baselining is needed) ──
    if importance_reweighting_method != "none" or baseline != "none" or advantage_normalizer != "none":
        # Use the standalone group-normalizer from this adapter
        advantages_t, _ = run_compute_group_normalized_rewards(
            raw_rewards=raw_rewards_t,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
        # advantages_t shape: (B,)
    else:
        advantages_t = raw_rewards_t.clone()

    # Dimensions for broadcasting: (B,) → (B, 1)
    raw_rewards_t = raw_rewards_t.unsqueeze(-1)   # (B, 1)
    advantages_t = advantages_t.unsqueeze(-1)      # (B, 1)

    # ── 4. Prepare old_log_probs (for off-policy methods) ──
    if old_log_probs is not None:
        old_log_probs = old_log_probs.detach()

    # ── 5. Microbatch loop ──
    micro_batch_size = B // gradient_accumulation_steps
    assert B % gradient_accumulation_steps == 0, \
        f"Batch size ({B}) must be divisible by gradient_accumulation_steps ({gradient_accumulation_steps})"

    total_loss = 0.0
    total_clip_frac = 0.0
    total_ratio = 0.0

    for micro_step in range(gradient_accumulation_steps):
        start = micro_step * micro_batch_size
        end = (micro_step + 1) * micro_batch_size

        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_mask = response_mask[start:end]
        mb_raw = raw_rewards_t[start:end]
        mb_adv = advantages_t[start:end]
        mb_old = old_log_probs[start:end] if old_log_probs is not None else None

        # Forward pass to get current log-probs
        log_probs_dict = get_response_log_probs(model, mb_input_ids, mb_labels)
        policy_log_probs = log_probs_dict["log_probs"]  # (micro_bs, L)

        # --- 5a. Compute per-token loss ---
        # For all methods, the weight is the advantage (which may be
        # raw_rewards if baseline="none" and normalizer="none").
        per_token_loss, meta = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_adv,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old,
            cliprange=cliprange,
            response_mask=mb_mask,
        )

        # --- 5b. Aggregate (scalar loss) ---
        microbatch_loss = run_aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mb_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        # --- 5c. Scale for gradient accumulation ---
        # - "sequence": each microbatch loss is mean(per-seq), need 1/K so K
        #   microbatches average to the global mean.
        # - "constant": each microbatch loss is total/C. The constant C is
        #   already a global normalizer, so no per-microbatch scaling needed.
        if loss_normalization == "constant":
            scaled_loss = microbatch_loss
        else:
            scaled_loss = microbatch_loss / gradient_accumulation_steps

        # --- 5d. Backward ---
        scaled_loss.backward()

        # Accumulate metrics
        total_loss += microbatch_loss.detach().item()
        total_clip_frac += meta.get("clip_fraction", torch.tensor(0.0)).detach().item()
        total_ratio += meta.get("ratio_mean", torch.tensor(1.0)).detach().item()

    # ── 6. Gradient clipping + optimizer step ──
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.tensor(0.0)

    optimizer.step()
    optimizer.zero_grad()

    # ── 7. Return ──
    # For "constant" normalization, total_loss is already the full-batch loss
    # (each microbatch reported total_micro/C, summed gives total_all/C).
    # For "sequence" normalization, total_loss is sum of per-microbatch means,
    # so we divide by K to get the global mean.
    if loss_normalization == "constant":
        avg_loss = total_loss
    else:
        avg_loss = total_loss / gradient_accumulation_steps
    metadata: dict[str, torch.Tensor | float] = {
        "loss": avg_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        "clip_fraction": total_clip_frac / gradient_accumulation_steps,
        "ratio_mean": total_ratio / gradient_accumulation_steps,
    }

    return torch.tensor(avg_loss), metadata


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    from cs336_alignment.core.batching import get_packed_sft_dataset as _impl
    return _impl(tokenizer=tokenizer, dataset_path=dataset_path, seq_length=seq_length, shuffle=shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    from cs336_alignment.core.batching import iterate_batches
    return iterate_batches(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    from cs336_alignment.eval.parsers import parse_mmlu_response
    return parse_mmlu_response(mmlu_example=mmlu_example, model_output=model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    from cs336_alignment.eval.parsers import parse_gsm8k_response
    return parse_gsm8k_response(model_output=model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
