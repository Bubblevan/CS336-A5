#!/usr/bin/env python3
"""distill_math.py

Distill reasoning traces from MATH train.jsonl using DeepSeek API (or compatible).

动机：
  MATH train 有 7,500 道题（problem + answer），但没有推理链。
  我们需要让 DeepSeek 为每道题生成完整的 <think>...</think> <answer>...</answer> 推理轨迹，
  作为 SFT 训练数据。

Pipeline:
  1. 加载 MATH train.jsonl → 提取 problem + answer（ground_truth）
  2. 用 r1_zero.prompt 模板格式化
  3. 调用 LLM API 生成推理轨迹
  4. 校验答案是否匹配 ground_truth（答错则跳过，防止小模型学错误推理）
  5. 保存为 SFT JSONL（prompt + response 对）

Usage:
    # 试跑 50 条
    uv run python scripts/distill_math.py --max_samples 50 --max_workers 20

    # 全量蒸馏（支持 --resume 断点续传）
    uv run python scripts/distill_math.py --max_workers 100
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
MATH_TRAIN_SRC = Path("/root/gpufree-share/data/MATH/train.jsonl")
OUTPUT_DIR     = Path("/root/gpufree-share/data/sft/math-distilled")
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
MAX_WORKERS = 100
file_lock = threading.Lock()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def extract_answer_from_response(text: str) -> str | None:
    """Extract answer from <answer>...</answer> tags in model response."""
    if not text:
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison: strip LaTeX whitespace, remove \\boxed{}."""
    ans = ans.strip()
    # Remove \boxed{}
    m = re.search(r"\\boxed\{([^}]*)\}", ans)
    if m:
        ans = m.group(1)
    # Remove trailing units like \text{ miles}
    ans = re.sub(r"\\text\{[^}]*\}$", "", ans).strip()
    # Remove leading/trailing $, \, whitespace
    ans = ans.replace("$", "").strip()
    # Remove \displaystyle
    ans = ans.replace("\\displaystyle", "")
    return ans.strip()


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches ground truth by normalized string + number equality."""
    if not pred or not gold:
        return False
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    # Try numeric comparison
    try:
        pred_clean = re.sub(r"[^0-9.\-/]", "", pred_norm)
        gold_clean = re.sub(r"[^0-9.\-/]", "", gold_norm)
        if pred_clean and gold_clean:
            import sympy
            if sympy.simplify(f"({pred_clean}) - ({gold_clean})") == 0:
                return True
    except Exception:
        pass
    return False


def call_deepseek(prompt: str, problem: str, ground_truth: str) -> dict | None:
    """Call DeepSeek API, verify answer, return SFT record or None."""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        content = response.choices[0].message.content.strip()

        # The model should continue from "Assistant: <think>"
        full_response = content

        # Extract answer and verify
        model_answer = extract_answer_from_response(full_response)
        if model_answer and answers_match(model_answer, ground_truth):
            return {
                "question": problem,
                "answer": ground_truth,
                "prompt": prompt,
                "response": full_response,
                "is_correct": True,
            }
        else:
            # Wrong answer—don't save
            return None

    except Exception as e:
        print(f"  API error: {e}")
        return None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distill MATH train.jsonl reasoning traces")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (default: all 7500)")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help="Concurrent API calls")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-processed problems)")
    args = parser.parse_args()

    # ── 1. Load MATH train.jsonl ──
    print(f"Loading MATH train from {MATH_TRAIN_SRC}...")
    with open(MATH_TRAIN_SRC) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.max_samples and args.max_samples < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, args.max_samples)
    print(f"  {len(rows)} samples loaded")

    # ── 2. Build task list ──
    tasks = []
    for row in rows:
        problem = row["problem"]
        ground_truth = str(row["answer"]).strip()
        tasks.append({
            "question": problem,
            "ground_truth": ground_truth,
            "prompt": PROMPT_TEMPLATE.format(question=problem),
        })
    print(f"  Built {len(tasks)} tasks")

    # ── 3. Resume support ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"

    processed_problems: set = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    processed_problems.add(json.loads(line)["question"])
                except Exception:
                    pass
        tasks = [t for t in tasks if t["question"] not in processed_problems]
        print(f"  Resuming: {len(processed_problems)} already done, {len(tasks)} remaining")
    else:
        output_path.write_text("")

    success = 0
    wrong = 0

    print(f"\nStarting distillation (workers={args.max_workers})...")

    def process_one(task):
        problem = task["question"]
        ground_truth = task["ground_truth"]
        api_prompt = task["prompt"] + "\n"

        result = call_deepseek(api_prompt, problem, ground_truth)
        if result is None:
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

    # ── 4. Report ──
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

    # ── 5. Symlink ──
    project_link = Path("/root/gpufree-data/cs336/data/math-distilled")
    if not project_link.exists():
        rel = Path("../../../gpufree-share/data/sft/math-distilled")
        project_link.symlink_to(rel)
        print(f"  Symlink: {project_link} → {rel}")
    else:
        print(f"  Symlink exists: {project_link}")


if __name__ == "__main__":
    main()
