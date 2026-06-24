#!/usr/bin/env python3
"""evaluate_tulu_baseline.py

Evaluate a base model (before SFT) on the TULU-3 SFT Personas Math dataset.

Usage:
    # HF backend (slow but reliable)
    uv run python scripts/evaluate_tulu_baseline.py \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --limit 200 --device cuda:0

    # vLLM backend (fast)
    uv run python scripts/evaluate_tulu_baseline.py \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --engine vllm --vllm_device cuda:0 --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from cs336_alignment.core.utils import get_model_and_tokenizer
from cs336_alignment.eval.generation import (
    GenerationConfig,
    VLLMGenerationConfig,
)
from cs336_alignment.eval.parsers import (
    parse_last_number,
    numbers_equal,
)


# ──────────────────────────────────────────────
# Answer extraction from TULU responses
# ──────────────────────────────────────────────

def extract_tulu_answer(text: str) -> str | None:
    """Extract the final answer from a TULU assistant response.

    TULU responses consistently end with:
        Final Answer: ... I hope it is correct.
    or sometimes just the final LaTeX expression.

    Strategy:
        1. Try "Final Answer:" → take everything after it (trim trailing fluff)
        2. Try \boxed{}  → extract boxed content
        3. Fallback → last number in the text
    """
    if not text:
        return None

    # Strategy 1: "Final Answer:" delimiter
    m = re.search(
        r'Final Answer:\s*(.+?)(?:\.\s*I hope it is correct\.|\.\s*$|$)',
        text,
        re.DOTALL,
    )
    if m:
        answer = m.group(1).strip().rstrip('.')
        # Clean up LaTeX delimiters
        answer = _clean_latex(answer)
        if answer:
            return answer

    # Strategy 2: \boxed{} (rare in TULU but handle anyway)
    if '\\boxed' in text:
        m_boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if m_boxed:
            return _clean_latex(m_boxed.group(1))

    # Strategy 3: last number
    import re as _re
    numbers = _re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if numbers:
        return numbers[-1]

    return None


def _clean_latex(s: str) -> str:
    """Strip LaTeX math delimiters from an answer string."""
    s = re.sub(r'^\\\[|\\\]$', '', s.strip())
    s = re.sub(r'^\\\(|\\\)$', '', s.strip())
    s = re.sub(r'^\$\$|\$\$$', '', s.strip())
    s = re.sub(r'^\$|\$$', '', s.strip())
    return s.strip()


def is_answer_correct(pred: str | None, gold: str | None) -> bool:
    """Compare predicted and gold answers.

    Strategy:
      1. Extract last number from both using parse_last_number
      2. Compare numerically with numbers_equal
      3. Fallback: normalized string equality (for non-numeric answers)
    """
    if pred is None or gold is None:
        return False

    # Try numeric comparison via last-number extraction
    pred_num = parse_last_number(pred)
    gold_num = parse_last_number(gold)
    if pred_num is not None and gold_num is not None:
        if numbers_equal(pred_num, gold_num):
            return True

    # Fallback: clean LaTeX noise and compare literally
    pred_str = _clean_latex(pred).strip().lower().rstrip('.')
    gold_str = _clean_latex(gold).strip().lower().rstrip('.')
    return pred_str == gold_str


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_tulu_data(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load TULU data and pre-extract ground truth answers."""
    with open(path) as f:
        rows = json.load(f)

    if limit and limit < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, limit)

    out = []
    skipped = 0
    for r in rows:
        gold = extract_tulu_answer(r['response'])
        if gold is None:
            skipped += 1
            continue
        out.append({
            "prompt":   r['prompt'],
            "response": r['response'],
            "gold":     gold,
        })

    if skipped:
        print(f"  (skipped {skipped} rows where gold answer could not be extracted)")

    return out


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    # 1. Load data
    print(f"\nLoading TULU data from {args.data_path}...")
    examples = load_tulu_data(args.data_path, limit=args.limit)
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

    for i, (prompt, gold, output) in enumerate(
        tqdm(zip(prompts, golds, model_outputs), total=len(examples), desc="Scoring")
    ):
        pred = extract_tulu_answer(output)
        is_correct = is_answer_correct(pred, gold)
        correct += int(is_correct)
        total += 1

        results.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "prompt": prompt[:200],
            "model_output": output[:500],
        })

    # 5. Report
    accuracy = correct / total if total > 0 else 0.0
    examples_per_sec = total / elapsed if elapsed > 0 else 0.0
    pred_rate = sum(1 for r in results if r["pred"] is not None) / total

    print(f"\n{'='*50}")
    print(f"TULU-3 Baseline Evaluation")
    print(f"{'='*50}")
    print(f"  Model:        {args.model_id}")
    print(f"  Samples:      {total}")
    print(f"  Accuracy:     {accuracy:.4f}  ({correct}/{total})")
    print(f"  Parse rate:   {pred_rate:.3f}  (model output → answer extraction rate)")
    print(f"  Time:         {elapsed:.1f}s  ({examples_per_sec:.1f} examples/s)")
    print(f"{'='*50}\n")

    # 6. Save results
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model_id,
                "num_samples": total,
                "accuracy": accuracy,
                "parse_rate": pred_rate,
                "elapsed_seconds": round(elapsed, 1),
                "results": results[: args.save_full_results],  # save first N for inspection
            }, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TULU-3 baseline evaluation")
    # Model
    parser.add_argument("--model_id", type=str,
                        default="/root/gpufree-share/models/Qwen2.5-Math-1.5B")
    # Data
    parser.add_argument("--data_path", type=str,
                        default="/root/gpufree-data/cs336/data/tulu-3-sft-personas-math/train.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of examples to evaluate (random sample)")
    # Engine
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hf_batch_size", type=int, default=8)
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--stop_strings", type=str, nargs="*", default=None)
    # Output
    parser.add_argument("--output_path", type=str,
                        default="outputs/tulu_baseline.json")
    parser.add_argument("--save_full_results", type=int, default=50)
    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
