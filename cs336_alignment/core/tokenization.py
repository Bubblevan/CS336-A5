"""
core/tokenization.py

tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    → dict["input_ids", "labels", "response_mask"]
    Tokenize prompt+response pairs, produce labels (shifted) and response mask.

apply_chat_template(messages, tokenizer) → str
    Apply tokenizer's chat template and return the formatted string.

pad_to_max_len(token_ids, pad_token_id, max_len) → torch.Tensor
"""
