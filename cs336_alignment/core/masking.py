"""
core/masking.py

masked_mean(tensor, mask, dim=None) → torch.Tensor
    Mean over unmasked positions.

masked_normalize(tensor, mask, normalize_constant, dim=None) → torch.Tensor
    Sum over unmasked positions divided by a constant.

build_response_mask(input_ids, prompt_lengths) → torch.Tensor
    Construct boolean mask: True for response tokens, False for prompt/padding.

make_causal_mask(seq_len, device) → torch.Tensor
"""
