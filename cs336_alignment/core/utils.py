"""Shared infrastructure utilities.

V1 更新：
- 更稳的 HuggingFace model/tokenizer loader；
- 支持 flash_attention_2；
- decoder-only generation 默认 left padding；
- 保留 JSON / JSONL 工具函数。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype is None or dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    normalized = dtype.lower()

    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype}")


def get_model_and_tokenizer(
    model_id_or_dir: str,
    device: str = "cuda:0",
    *,
    dtype: str | torch.dtype | None = "bfloat16",
    attn_implementation: str | None = "flash_attention_2",
    trust_remote_code: bool = True,
):
    """Load a HuggingFace causal LM and tokenizer.

    Args:
        model_id_or_dir:
            HuggingFace model id or local checkpoint path.
        device:
            "cuda:0", "cuda:1", or "cpu".
        dtype:
            "bfloat16", "float16", "float32", "auto", or torch.dtype.
        attn_implementation:
            For CUDA, usually "flash_attention_2" if installed.
            Use "eager" or None if debugging compatibility issues.
        trust_remote_code:
            Passed to HuggingFace loaders.

    Returns:
        model, tokenizer
    """
    device_obj = torch.device(device)
    torch_dtype = resolve_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_dir,
        trust_remote_code=trust_remote_code,
    )

    # Decoder-only generation 通常建议 left padding。
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if attn_implementation is not None:
        # CPU 上 flash attention 不可用，自动降级。
        if device_obj.type == "cpu" and attn_implementation == "flash_attention_2":
            model_kwargs["attn_implementation"] = "eager"
        else:
            model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        **model_kwargs,
    )

    model.to(device_obj)
    model.eval()

    return model, tokenizer


def json_safe(obj: Any) -> Any:
    """Convert common Python / torch / numpy objects to JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]

    return obj


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(obj), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}") from e

    return rows


def write_json(path: str | Path, obj: Any) -> None:
    """Write one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)