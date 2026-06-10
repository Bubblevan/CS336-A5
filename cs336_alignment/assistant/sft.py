"""
assistant/sft.py

assistant_sft_train_step(policy_log_probs, response_mask,
                         grad_accum_steps, normalize_constant)
    → tuple[loss, metadata]
    SFT micro-batch step for assistant-tuning (same core, different packaging).

log_assistant_generations(...)
    Generation logging specific to safety / instruction-following eval.
"""
