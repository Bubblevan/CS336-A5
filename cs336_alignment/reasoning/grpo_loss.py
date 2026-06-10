"""
reasoning/grpo_loss.py

compute_naive_policy_gradient_loss(raw_rewards_or_advantages,
                                   policy_log_probs)
    → Tensor

compute_grpo_clip_loss(advantages, policy_log_probs,
                       old_log_probs, cliprange)
    → tuple[loss: Tensor, metadata: dict]

compute_policy_gradient_loss(policy_log_probs, loss_type,
                             raw_rewards=None, advantages=None,
                             old_log_probs=None, cliprange=None)
    → tuple[loss: Tensor, metadata: dict]
    Unified dispatch: "no_baseline" | "reinforce_with_baseline" | "grpo_clip".
"""
