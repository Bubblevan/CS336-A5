"""
reasoning/sft.py

sft_microbatch_train_step(policy_log_probs, response_mask,
                          grad_accum_steps, normalize_constant=1.0)
    → tuple[loss: Tensor, metadata: dict]
    Compute SFT cross-entropy loss over response tokens, backprop.

log_generations(prompts, ground_truths, vllm_model, policy_model,
                tokenizer, sampling_params, reward_fn,
                save_jsonl_path=None, save_metrics_path=None)
    → tuple[records: list[dict], metrics: dict]
    Generate responses, score them, log stats.
"""
