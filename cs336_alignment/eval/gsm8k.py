"""GSM8K baseline evaluation.

V1 更新：
- 生成逻辑从 eval/gsm8k.py 中拆出；
- 支持 HFGenerator / VLLMGenerator；
- 一次构造所有 prompts，然后批量生成；
- 输出 summary + predictions.jsonl；
- 记录 eval wall time 和 examples/sec。

暂时不做：
- few-shot prompting
- self-consistency
- pass@k / majority vote
- answer-level normalization beyond numeric matching
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cs336_alignment.core.utils import append_jsonl, load_jsonl
from cs336_alignment.eval.generation import TextGenerator
from cs336_alignment.eval.parsers import (
    numbers_equal,
    parse_gsm8k_gold_answer,
    parse_gsm8k_response,
)


def make_gsm8k_prompt(question: str) -> str:
    """Build a simple zero-shot GSM8K prompt.

    这个 prompt 保持和上一版一致，保证 baseline 可比。
    后续可以新增 --prompt_style 来比较：
    - zero_shot
    - few_shot
    - qwen_math_style
    - socratic
    """
    return (
        "Solve the following grade school math problem. "
        "Reason step by step, and put only the final numeric answer "
        "inside <answer>...</answer>.\n\n"
        f"Problem:\n{question}\n\n"
        "Solution:\n"
    )


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read parquet using pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires pyarrow. "
            "Install it or convert GSM8K parquet to jsonl first."
        ) from e

    table = pq.read_table(path)
    return table.to_pylist()


def _find_gsm8k_file(path: Path, split: str) -> Path:
    """Resolve a GSM8K file path.

    Supported inputs:
    1. data/gsm8k/main/test-00000-of-00001.parquet
    2. data/gsm8k/main
    3. data/gsm8k

    If data/gsm8k is given, prefer data/gsm8k/main/{split}-*.parquet.
    """
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"GSM8K path does not exist: {path}")

    candidates: list[Path] = []

    main_dir = path / "main"
    if main_dir.is_dir():
        candidates.extend(sorted(main_dir.glob(f"{split}-*.parquet")))
        candidates.extend(sorted(main_dir.glob(f"{split}.jsonl")))

    candidates.extend(sorted(path.glob(f"{split}-*.parquet")))
    candidates.extend(sorted(path.glob(f"{split}.jsonl")))

    candidates.extend(sorted(path.glob("*.parquet")))
    candidates.extend(sorted(path.glob("*.jsonl")))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find GSM8K {split} parquet/jsonl under: {path}"
        )

    return candidates[0]


def load_gsm8k_examples(
    data_path: str | Path,
    *,
    split: str = "test",
) -> list[dict[str, str | None]]:
    """Load GSM8K examples from parquet or jsonl."""
    path = _find_gsm8k_file(Path(data_path), split=split)

    if path.suffix == ".parquet":
        raw_rows = _read_parquet(path)
    elif path.suffix == ".jsonl":
        raw_rows = load_jsonl(path)
    else:
        raise ValueError(f"Unsupported GSM8K file format: {path}")

    examples: list[dict[str, str | None]] = []

    for row in raw_rows:
        question = row.get("question")
        answer = row.get("answer")

        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        gold = parse_gsm8k_gold_answer(answer)

        examples.append(
            {
                "question": question,
                "answer": answer,
                "gold": gold,
            }
        )

    return examples


def run_gsm8k_eval(
    generator: TextGenerator,
    gsm8k_data_path: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    max_new_tokens: int = 512,
    output_path: str | Path | None = None,
    stop_strings: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a model/backend on GSM8K.

    Args:
        generator:
            HFGenerator or VLLMGenerator.
        gsm8k_data_path:
            Path to GSM8K parquet/jsonl file or directory.
        split:
            Usually "test".
        limit:
            If set, only evaluate the first N examples.
        max_new_tokens:
            Generation budget.
        output_path:
            If set, write per-example predictions as JSONL.
        stop_strings:
            Optional generation stop strings.

    Returns:
        Summary dict.
    """
    examples = load_gsm8k_examples(gsm8k_data_path, split=split)

    if limit is not None and limit > 0:
        examples = examples[:limit]

    if not examples:
        raise RuntimeError(f"No GSM8K examples found at {gsm8k_data_path}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    prompts = [
        make_gsm8k_prompt(str(example["question"]))
        for example in examples
    ]

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

    total = 0
    correct = 0
    parsed = 0
    gold_parsed = 0

    for idx, (example, prompt, model_output) in enumerate(
        tqdm(
            zip(examples, prompts, model_outputs, strict=True),
            total=len(examples),
            desc="GSM8K parse/score",
        )
    ):
        gold = example["gold"]
        pred = parse_gsm8k_response(model_output)
        is_correct = numbers_equal(pred, gold)

        total += 1
        correct += int(is_correct)
        parsed += int(pred is not None)
        gold_parsed += int(gold is not None)

        if output_path is not None:
            append_jsonl(
                output_path,
                {
                    "idx": idx,
                    "question": example["question"],
                    "gold_answer_raw": example["answer"],
                    "gold": gold,
                    "prompt": prompt,
                    "model_output": model_output,
                    "pred": pred,
                    "correct": is_correct,
                },
            )

    accuracy = correct / total if total > 0 else 0.0
    parse_rate = parsed / total if total > 0 else 0.0
    gold_parse_rate = gold_parsed / total if total > 0 else 0.0
    examples_per_second = total / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return {
        "benchmark": "gsm8k",
        "split": split,
        "data_path": str(gsm8k_data_path),
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "gold_parse_rate": gold_parse_rate,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": examples_per_second,
    }