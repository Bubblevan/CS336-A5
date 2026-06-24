"""MATH baseline evaluation.

V0 目标：
- 在 MATH validation set 上评估 base model 的数学推理能力；
- 使用 r1_zero.prompt 模板（与 SFT/GRPO 训练一致的 prompt 格式）；
- 使用 reasoning/rewards.py 的 grade() 做 LaTeX-aware 判分；
- 复用 TextGenerator Protocol，支持 HF / vLLM 后端。

为什么需要这个文件？
- GSM8K 的 parse_last_number 只能处理纯数字答案，MATH 答案用 LaTeX 表示；
- MATH 需要符号级等价性判断（\frac{1}{9} ≡ 1/9）；
- 但不需重写生成后端——TextGenerator Protocol 已在 eval/generation.py 中抽象好。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cs336_alignment.core.utils import append_jsonl, load_jsonl
from cs336_alignment.eval.generation import TextGenerator
from cs336_alignment.reasoning.rewards import grade


# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────

_R1_ZERO_TEMPLATE: str | None = None


def _load_r1_zero_template() -> str:
    """Lazy-load the r1_zero.prompt template."""
    global _R1_ZERO_TEMPLATE
    if _R1_ZERO_TEMPLATE is not None:
        return _R1_ZERO_TEMPLATE

    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "r1_zero.prompt"
    _R1_ZERO_TEMPLATE = prompt_path.read_text(encoding="utf-8")
    return _R1_ZERO_TEMPLATE


def make_math_prompt(problem: str) -> str:
    """Build a prompt for MATH evaluation using the r1_zero template."""
    template = _load_r1_zero_template()
    return template.format(question=problem)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_math_examples(
    data_path: str | Path,
    *,
    split: str = "validation",
    limit: int | None = None,
) -> list[dict[str, str | None]]:
    """Load MATH examples from a JSONL file.

    MATH JSONL format (one JSON object per line):
        {"problem": "...", "level": "Level 3", "subject": "Prealgebra",
         "unique_id": "test/prealgebra/126.json", "answer": "420"}

    Args:
        data_path: Path to MATH JSONL file (e.g. validation.jsonl).
        split: Ignored; kept for API consistency with GSM8K eval.
        limit: If set, only load the first N examples.

    Returns:
        List of dicts with keys: problem, answer_raw (ground truth LaTeX).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"MATH data file not found: {path}")

    raw_rows = load_jsonl(path)

    examples: list[dict[str, str | None]] = []
    for idx, row in enumerate(raw_rows):
        problem = row.get("problem")
        answer = row.get("answer")

        if not isinstance(problem, str) or not isinstance(answer, str):
            continue

        examples.append({
            "idx": idx,
            "problem": problem,
            "answer_raw": answer,       # ground truth LaTeX, e.g. "\dfrac{1}{9}"
            "level": row.get("level"),
            "subject": row.get("subject"),
        })

        if limit is not None and len(examples) >= limit:
            break

    return examples


# ──────────────────────────────────────────────
# Answer extraction from model output
# ──────────────────────────────────────────────

def extract_math_answer(model_output: str) -> str | None:
    """Extract the answer from a model's response.

    Expected format (r1_zero style):
        <think> reasoning... </think> <answer> 42 </answer>

    Strategy:
        1. Try <answer>…</answer> tags → extract inner content
        2. Fallback → None (unformatted output = wrong)

    Returns:
        Extracted answer string (may be LaTeX), or None if not found.
    """
    if not model_output:
        return None

    text = str(model_output)

    # Try <answer> tags first (r1_zero format)
    import re
    tag_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_match:
        raw = tag_match.group(1).strip()
        return raw if raw else None

    # No valid format found
    return None


# ──────────────────────────────────────────────
# Evaluation orchestrator
# ──────────────────────────────────────────────

def run_math_eval(
    generator: TextGenerator,
    math_data_path: str | Path,
    *,
    split: str = "validation",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    output_path: str | Path | None = None,
    stop_strings: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a model on the MATH validation set.

    Args:
        generator: HFGenerator or VLLMGenerator.
        math_data_path: Path to MATH JSONL file (e.g. validation.jsonl).
        split: Ignored; kept for API consistency.
        limit: Evaluate only first N examples.
        max_new_tokens: Generation budget per example.
        output_path: If set, write per-example predictions as JSONL.
        stop_strings: Optional generation stop strings.

    Returns:
        Summary dict with accuracy, parse rate, etc.
    """
    examples = load_math_examples(math_data_path, split=split, limit=limit)

    if not examples:
        raise RuntimeError(f"No MATH examples found at {math_data_path}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    # Build prompts
    problems = [ex["problem"] for ex in examples]
    prompts = [make_math_prompt(p) for p in problems]

    # Generate
    start_time = time.perf_counter()

    model_outputs = generator.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        stop_strings=stop_strings,
    )

    elapsed_seconds = time.perf_counter() - start_time

    if len(model_outputs) != len(examples):
        raise RuntimeError(
            f"Generator returned {len(model_outputs)} outputs for "
            f"{len(examples)} prompts."
        )

    # Score
    total = 0
    correct = 0
    formatted = 0          # outputs that followed <answer> format
    by_subject: dict[str, dict[str, int]] = {}
    by_level: dict[str, dict[str, int]] = {}

    for idx, (example, prompt, model_output) in enumerate(
        tqdm(
            zip(examples, prompts, model_outputs, strict=True),
            total=len(examples),
            desc="MATH parse/score",
        )
    ):
        gold = example["answer_raw"]
        pred = extract_math_answer(model_output)

        # Use the same grade() function that GRPO training uses
        is_correct = False
        if pred is not None:
            formatted += 1
            is_correct = grade(pred, gold, fast=True)

        total += 1
        correct += int(is_correct)

        # Per-subject and per-level breakdown
        subject = example.get("subject") or "unknown"
        level = example.get("level") or "unknown"

        for bucket, key in [(by_subject, subject), (by_level, level)]:
            if key not in bucket:
                bucket[key] = {"total": 0, "correct": 0}
            bucket[key]["total"] += 1
            bucket[key]["correct"] += int(is_correct)

        if output_path is not None:
            append_jsonl(
                output_path,
                {
                    "idx": idx,
                    "problem": example["problem"],
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "level": example.get("level"),
                    "subject": example.get("subject"),
                },
            )

    accuracy = correct / total if total > 0 else 0.0
    format_rate = formatted / total if total > 0 else 0.0
    examples_per_second = total / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return {
        "benchmark": "math",
        "split": split,
        "data_path": str(math_data_path),
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": examples_per_second,
        "by_subject": {
            subj: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
            for subj, stats in sorted(by_subject.items())
        },
        "by_level": {
            lvl: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
            for lvl, stats in sorted(by_level.items())
        },
    }
