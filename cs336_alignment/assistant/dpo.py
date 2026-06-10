"""
assistant/dpo.py

compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta,
                              prompt, response_chosen, response_rejected)
    → Tensor
    DPO loss for a single preference pair.

compute_batch_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                       ref_chosen_logps, ref_rejected_logps, beta)
    → Tensor
    Batched DPO loss from pre-computed log-probs.
"""
