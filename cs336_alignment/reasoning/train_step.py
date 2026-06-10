"""
reasoning/train_step.py

grpo_microbatch_train_step(policy_log_probs, response_mask,
                           grad_accum_steps, loss_type,
                           raw_rewards, advantages,
                           old_log_probs, cliprange)
    → tuple[loss: Tensor, metadata: dict]
    Full GRPO microbatch: compute PG loss + backprop.
"""
