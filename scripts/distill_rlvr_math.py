#!/usr/bin/env python3
"""distill_rlvr_math.py

Distill reasoning traces from RLVR-MATH using DeepSeek API (or compatible).

Pipeline:
  1. Load RLVR-MATH parquet → extract target question + ground_truth
  2. Format with r1_zero.prompt template
  3. Call LLM API to generate reasoning trace (<think>...</think> <answer>...</answer>)
  4. Verify the answer matches ground_truth
  5. Save verified (prompt, response) pairs as SFT JSONL

Usage:
    uv run python scripts/distill_rlvr_math.py \
        --max_samples 100 \
        --max_workers 50
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY      = os.environ["LLM_API_KEY"]
BASE_URL     = os.environ["LLM_BASE_URL"]
MODEL_ID     = os.environ["LLM_MODEL_ID"]

# File paths
RLVR_MATH_SRC = Path("/root/gpufree-share/data/RLVR-MATH/data/train-00000-of-00001.parquet")
OUTPUT_DIR    = Path("/root/gpufree-share/data/sft/rlvr-math-distilled")
PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the "
    "reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

# Concurrency
MAX_WORKERS = 50
file_lock = threading.Lock()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ──────────────────────────────────────────────
# Extract target question from RLVR-MATH
# ──────────────────────────────────────────────

def extract_target_question(messages_content: str) -> str | None:
    """Extract the LAST question from a few-shot RLVR-MATH prompt.

    The format is:
        Question: ... Answer: \boxed{...}
        Question: ... Answer: \boxed{...}
        ...
        Question: (TARGET — no answer follows)
    """
    # Split by "Question:" and take the last segment
    parts = messages_content.split("Question:")
    if len(parts) < 2:
        return None
    last_part = parts[-1].strip()
    # Remove any trailing "Answer:" if present (shouldn't be, but guard)
    last_question = re.split(r'\n\s*Answer:', last_part)[0].strip()
    return last_question


# ──────────────────────────────────────────────
# Answer extraction from model output
# ──────────────────────────────────────────────

def extract_answer_from_response(text: str) -> str | None:
    """Extract the answer from <answer> tags or \\boxed{} or last number."""
    # Strategy 1: <answer>...</answer>
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strategy 2: \boxed{}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer for comparison."""
    # Strip LaTeX delimiters
    ans = re.sub(r'^\\\(|\\\)$', '', ans.strip())
    ans = re.sub(r'^\$|\$$', '', ans.strip())
    # Remove whitespace
    ans = re.sub(r'\s+', '', ans)
    return ans


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches ground truth."""
    return normalize_answer(pred) == normalize_answer(gold)


# ──────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────

def call_deepseek(prompt: str, question: str, ground_truth: str) -> dict | None:
    """Call DeepSeek API, validate answer, return SFT record or None."""
    max_retries = 5

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=2048,
                timeout=120,
            )

            content = completion.choices[0].message.content or ""
            reasoning = getattr(completion.choices[0].message, 'reasoning_content', None)
            reasoning = (reasoning or "").strip()

            # Build the full response: reasoning + answer
            # If the model already outputs <think>...</think> <answer>...</answer>, use as-is
            # Otherwise wrap it
            if "<think>" in content and "<answer>" in content:
                full_response = content.strip()
            elif reasoning:
                # Some models put reasoning in reasoning_content, answer in content
                full_response = f"<think> {reasoning} </think> {content.strip()}"
            else:
                full_response = f"<think> {content} </think>"

            # Extract answer for validation
            extracted = extract_answer_from_response(content)
            if extracted is None:
                extracted = extract_answer_from_response(full_response)

            if extracted and answers_match(extracted, ground_truth):
                # Verified correct → save for SFT
                sft_record = {
                    "question": question,
                    "answer": ground_truth,
                    "prompt": PROMPT_TEMPLATE.format(question=question),
                    "response": full_response,
                    "is_correct": True,
                }
                return sft_record
            else:
                # Answer doesn't match → skip (don't train on wrong answers)
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  API failed after {max_retries} attempts: {e}")
            return None

    return None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distill RLVR-MATH reasoning traces")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (default: all 7500)")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help="Concurrent API calls")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-processed questions)")
    args = parser.parse_args()

    # ── 1. Load data ──
    print(f"Loading RLVR-MATH from {RLVR_MATH_SRC}...")
    table = pq.read_table(str(RLVR_MATH_SRC))
    rows = table.to_pylist()
    if args.max_samples and args.max_samples < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, args.max_samples)
    print(f"  {len(rows)} samples loaded")

    # ── 2. Build task list ──
    tasks = []
    for row in rows:
        msg_content = row["messages"][0]["content"]
        target_q = extract_target_question(msg_content)
        if target_q is None:
            continue
        ground_truth = str(row["ground_truth"]).strip()
        tasks.append({
            "question": target_q,
            "ground_truth": ground_truth,
            "prompt": PROMPT_TEMPLATE.format(question=target_q),
        })
    print(f"  Extracted {len(tasks)} target questions")

    # ── 3. Resume support ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"

    processed_questions: set = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    processed_questions.add(json.loads(line)["question"])
                except Exception:
                    pass
        tasks = [t for t in tasks if t["question"] not in processed_questions]
        print(f"  Resuming: {len(processed_questions)} already done, {len(tasks)} remaining")
    else:
        # 非 resume 模式：清空文件从头写
        output_path.write_text("")
    success = 0
    failed = 0
    wrong = 0

    print(f"\nStarting distillation (workers={args.max_workers})...")

    def process_one(task):
        question = task["question"]
        ground_truth = task["ground_truth"]
        api_prompt = task["prompt"] + "\n"

        result = call_deepseek(api_prompt, question, ground_truth)
        if result is None:
            # Wrong answer or API error
            return None
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        pbar = tqdm(total=len(tasks), desc="Distilling")

        for future in as_completed(futures):
            result = future.result()
            if result:
                with file_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success += 1
            else:
                wrong += 1

            pbar.update(1)
            pbar.set_postfix({"correct": success, "skipped": wrong})

        pbar.close()

    # ── 5. Report ──
    print(f"\n{'='*50}")
    print(f"Done! Results in {output_path}")
    print(f"  Total processed:  {len(tasks)}")
    print(f"  Correct (saved):  {success}")
    print(f"  Skipped (wrong):  {wrong}")
    print(f"  Accuracy rate:    {success / max(len(tasks), 1):.1%}")
    print(f"{'='*50}")

    # Also write JSON for compatibility
    records = []
    with open(output_path) as f:
        for line in f:
            records.append(json.loads(line))
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Also wrote {json_path}")

    # ── 6. Symlink ──
    project_link = Path("/root/gpufree-data/cs336/data/rlvr-math-distilled")
    if not project_link.exists():
        rel = Path("../../../gpufree-share/data/sft/rlvr-math-distilled")
        project_link.symlink_to(rel)
        print(f"  Symlink: {project_link} → {rel}")
    else:
        print(f"  Symlink exists: {project_link}")


if __name__ == "__main__":
    main()
