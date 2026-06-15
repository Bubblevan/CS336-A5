"""Shared infrastructure utilities.

V0 目标：
- 稳定加载 HuggingFace causal LM + tokenizer；
- 设置随机种子；
- 提供最小 JSON / JSONL 工具函数。

后续再慢慢扩展：
- checkpoint resume
- run name
- wandb
- distributed training
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _get_attn_implementation(device: torch.device) -> str:
    """Pick attention backend: flash_attention_2 if available, else eager."""
    if device.type == "cpu":
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        print("Warning: flash-attn not installed, falling back to eager attention.")
        return "eager"

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_and_tokenizer(
    model_id_or_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    dtype: torch.dtype | None=None,
):
    """Load a HuggingFace model and tokenizer from a local path or hub id."""
    device_obj = torch.device(device)

    if dtype is None:
        dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir, trust_remote_code=True)
    # 一般来说 Decoder-only 模型的 tokenizer 都没有 pad token，所以我们需要手动设置一下。
    # 对于 Generation/eval，一般都重用 EOS token 作为 PAD token 就好了。
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=dtype,
        attn_implementation=_get_attn_implementation(device_obj),
    )

    model.to(device_obj)
    model.eval()

    return model, tokenizer

# --- JSON / JSONL utils ---
def json_safe(obj: Any) -> Any:
    """Convert an object to a JSON-serializable format."""
    # 对于 Path 对象，我们希望它能被转换成一个普通的字符串路径，这样就可以直接被 JSON 序列化了。
    if isinstance(obj, Path):
        return str(obj)
    # 对于 Tensor，我们希望它能被转换成一个普通的 Python 数字或者列表，这样就可以直接被 JSON 序列化了。
    if isinstance(obj, torch.Tensor):
        if obj.numel()==1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(json_safe(x) for x in obj)

    return obj

def append_jsonl(path: str |Path, obj: dict[str, Any]) -> None:
    """Append an object to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False)
        f.write("\n")

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    rows: list[dict[str, Any]] = [] # 先声明一个空列表，避免后续 mypy 报错 "Incompatible types in assignment (expression has type "list[dict[str, Any]]", variable has type "list[dict[str, Any]]")"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_idx} of {path}: {e}")
                continue
        return rows

def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Write an object to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)

