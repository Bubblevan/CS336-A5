"""
core/utils.py

Shared infrastructure utilities.

get_model_and_tokenizer(model_id_or_dir, device)
    Load a HuggingFace model + tokenizer.

set_seed(seed)
    Seed random, numpy, torch, torch.cuda.

json_safe(obj) → JSON-compatible object
append_jsonl / load_jsonl / load_json_list / write_json_list
resolve_output_path / make_run_name / save_checkpoint
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    """Load a HuggingFace model and tokenizer from a local path or hub id."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device == "cpu" else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
