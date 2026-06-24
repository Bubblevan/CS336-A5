#!/usr/bin/env python3
"""evaluate_rlvr_baseline.py

Evaluate a base model on RLVR-MATH (before SFT).

RLVR-MATH provides clean ground_truth for each question, so we just need to:
  1. Extract the target question from the few-shot messages
  2. Generate a response with the base model
  3. Extract the answer and compare to ground_truth

Usage:
    uv run python scripts/evaluate_rlvr_baseline.py \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --engine vllm --vllm_device cuda:0 --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

from cs336_alignment.core.utils import get_model_and_tokenizer
from cs336_alignment.eval.generation import (
    GenerationConfig,
    VLLMGenerationConfig,
)
from cs336_alignment.eval.parsers import parse_last_number, numbers_equal


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def extract_target_question(messages_content: str) -> str | None:
    """Extract the LAST question from RLVR-MATH's few-shot format."""
    parts = messages_content.split("Question:")
    if len(parts) < 2:
        return None
    last_part = parts[-1].strip()
    last_question = re.split(r'\n\s*Answer:', last_part)[0].strip()
    return last_question


def load_rlvr_data(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load RLVR-MATH parquet, extract target questions + ground_truth."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    rows = table.to_pylist()

    if limit and limit < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, limit)

    out = []
    skipped = 0
    for row in rows:
        msg_content = row["messages"][0]["content"]
        target_q = extract_target_question(msg_content)
        if target_q is None:
            skipped += 1
            continue
        out.append({
            "prompt": target_q,
            "gold": str(row["ground_truth"]).strip(),
        })

    if skipped:
        print(f"  (skipped {skipped} rows — could not extract target question)")

    return out


# ──────────────────────────────────────────────
# Answer extraction & scoring
# ──────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract answer from model output — try <answer>, \\boxed{}, last number."""
    # <answer> tag
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        # Try to get the last number from the answer (handles "the answer is 42" inside tags)
        num = parse_last_number(ans)
        if num:
            return num
        return ans

    # \boxed{}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()

    # Last number in the entire text
    num = parse_last_number(text)
    if num:
        return num

    return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    # Numerical comparison
    if numbers_equal(pred, gold):
        return True
    # String equality
    return pred.strip().lower() == gold.strip().lower()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    # 1. Load data
    print(f"\nLoading RLVR-MATH from {args.data_path}...")
    examples = load_rlvr_data(args.data_path, limit=args.limit)
    print(f"  {len(examples)} examples loaded")

    prompts = [ex["prompt"] for ex in examples]
    golds   = [ex["gold"]   for ex in examples]

    # 2. Build generator
    if args.engine == "vllm":
        print(f"Initializing vLLM on {args.vllm_device}...")
        generator = VLLMGenerationConfig(
            model_id_or_dir=args.model_id,
            dtype=args.dtype,
            gpu_memory_utilization=args.vllm_gpu_util,
            seed=args.seed,
            enable_prefix_caching=True,
        )
    else:
        print(f"Loading model {args.model_id} on {args.device}...")
        model, tokenizer = get_model_and_tokenizer(
            args.model_id,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        generator = GenerationConfig(
            model=model,
            tokenizer=tokenizer,
            device=torch.device(args.device),
            batch_size=args.hf_batch_size,
        )

    # 3. Generate
    print(f"Generating {len(prompts)} responses (max {args.max_new_tokens} tokens)...")
    start_time = time.perf_counter()
    model_outputs = generator.generate(
        prompts,
        max_new_tokens=args.max_new_tokens,
        stop_strings=args.stop_strings,
    )
    elapsed = time.perf_counter() - start_time

    # 4. Score
    correct = 0
    total = 0
    results = []

    for prompt, gold, output in tqdm(
        zip(prompts, golds, model_outputs), total=len(examples), desc="Scoring"
    ):
        pred = extract_answer(output)
        c = is_correct(pred, gold)
        correct += int(c)
        total += 1
        results.append({
            "prompt": prompt[:150],
            "gold": gold,
            "pred": pred,
            "correct": c,
        })

    accuracy = correct / total if total > 0 else 0.0
    parse_rate = sum(1 for r in results if r["pred"] is not None) / total
    examples_per_sec = total / elapsed if elapsed > 0 else 0.0

    # 5. Report
    print(f"\n{'='*50}")
    print(f"RLVR-MATH Baseline Evaluation")
    print(f"{'='*50}")
    print(f"  Model:        {args.model_id}")
    print(f"  Samples:      {total}")
    print(f"  Accuracy:     {accuracy:.4f}  ({correct}/{total})")
    print(f"  Parse rate:   {parse_rate:.3f}")
    print(f"  Time:         {elapsed:.1f}s  ({examples_per_sec:.1f} examples/s)")
    print(f"{'='*50}\n")

    # 6. Save
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model_id,
                "num_samples": total,
                "accuracy": accuracy,
                "parse_rate": parse_rate,
                "elapsed_seconds": round(elapsed, 1),
                "results": results[: args.save_full_results],
            }, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR-MATH baseline evaluation")
    parser.add_argument("--model_id", type=str,
                        default="/root/gpufree-share/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data_path", type=str,
                        default="/root/gpufree-share/data/RLVR-MATH/data/train-00000-of-00001.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hf_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--stop_strings", type=str, nargs="*", default=None)
    parser.add_argument("--output_path", type=str,
                        default="outputs/rlvr_baseline.json")
    parser.add_argument("--save_full_results", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
