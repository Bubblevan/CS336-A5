"""
reasoning/grpo_advantage.py

compute_group_normalized_rewards(reward_fn, rollout_responses,
                                 repeated_ground_truths,
                                 group_size, advantage_eps,
                                 normalize_by_std)
    → tuple[normalized_rewards: Tensor, raw_rewards: Tensor,
            metadata: dict]
    Compute per-group normalized advantages for GRPO.
"""
