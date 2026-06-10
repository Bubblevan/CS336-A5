"""
core/scoring.py

compute_entropy(logits) → torch.Tensor
    Per-token entropy of the predictive distribution.

get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
    → dict["log_probs": Tensor, "token_entropy"? : Tensor]
    Extract log-probabilities of ground-truth tokens from model outputs.

compute_log_probs_from_logits(logits, labels) → torch.Tensor
    Pure-tensor helper: gather label log-probs from a logits tensor.
"""
