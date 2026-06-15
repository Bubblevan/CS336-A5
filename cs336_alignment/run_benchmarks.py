"""Unified benchmark runner.

V0 只实现 GSM8K baseline。

Example:
    uv run python -m cs336_alignment.run_benchmarks \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --benchmarks gsm8k \
        --gsm8k_path data/gsm8k/main/test-00000-of-00001.parquet \
        --output_dir outputs/baseline_qwen_math \
        --device cuda:0 \
        --limit 20 \
        --max_new_tokens 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_alignment.core.utils import (
    get_model_and_tokenizer,
    set_seed,
    write_json,
)
from cs336_alignment.eval.gsm8k import run_gsm8k_eval

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k",
        help="Comma-separated benchmark names. V0 supports: gsm8k.",
    )
    parser.add_argument(
        "--gsm8k_path",
        type=str,
        default="data/gsm8k/main/test-00000-of-00001.parquet",
        help="Path to GSM8K parquet/jsonl file or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline",
        help="Directory for summary.json and predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N examples. Use for smoke tests.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Generation budget per example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_names = {
        name.strip().lower()
        for name in args.benchmarks.split(",")
        if name.strip()
    }

    unsupported = benchmark_names - {"gsm8k"}
    if unsupported:
        raise ValueError(
            f"Unsupported benchmarks in V0: {sorted(unsupported)}. "
            "Currently only 'gsm8k' is implemented."
        )

    print(f"Loading model and tokenizer from: {args.model_id}")
    model, tokenizer = get_model_and_tokenizer(
        args.model_id,
        device=args.device,
    )

    summaries = {}

    if "gsm8k" in benchmark_names:
        predictions_path = output_dir / "gsm8k_predictions.jsonl"

        print(f"Running GSM8K eval on: {args.gsm8k_path}")
        gsm8k_summary = run_gsm8k_eval(
            model=model,
            tokenizer=tokenizer,
            gsm8k_path=args.gsm8k_path,
            split="test",
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            output_path=predictions_path,
            device=args.device,
        )

        summaries["gsm8k"] = gsm8k_summary

        print("\nGSM8K summary:")
        for key, value in gsm8k_summary.items():
            print(f"  {key}: {value}")

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summaries)

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()