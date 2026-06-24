#!/usr/bin/env python3
"""prepare_sft_data.py

Prepare SFT training data from raw sources.

Converts:
  - GSM8K (parquet)          → JSONL with prompt/response for SFT
  - TULU-3 SFT Personas Math (parquet) → JSONL with prompt/response for SFT

Output format (JSONL, one object per line):
  {
    "question":   str,   # original question / problem text
    "answer":     str,   # original answer / gold (GSM8K) or empty (TULU)
    "prompt":     str,   # formatted prompt (ready to feed into model)
    "response":   str,   # expected SFT response
  }

Usage:
    uv run python scripts/prepare_sft_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

GSM8K_SRC  = Path("/root/gpufree-share/data/gsm8k")
TULU_SRC   = Path("/root/gpufree-share/data/tulu-3-sft-personas-math")
DATA_DST   = Path("/root/gpufree-share/data/sft")

R1_ZERO_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the "
    "reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


# ──────────────────────────────────────────────
# 1. GSM8K
# ──────────────────────────────────────────────

def _read_parquet(path: Path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _format_gsm8k_response(answer: str) -> str:
    """Turn GSM8K answer into r1-zero style response.

    GSM8K answer format:
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
         Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n
         #### 72"

    We extract the CoT reasoning and the final numeric answer, then wrap in
    <think>/<answer> tags.
    """
    if "####" in answer:
        parts = answer.rsplit("####", maxsplit=1)
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        reasoning = answer.strip()
        final_answer = answer.strip()

    # Clean up <<expr=result>> markers. GSM8K format: <<48/2=24>>24
    # We remove the <<...>> marker entirely and keep the trailing result as-is.
    import re
    reasoning = re.sub(r"<<[^>]+>>", "", reasoning)

    return f"<think> {reasoning} </think> <answer> {final_answer} </answer>"


def convert_gsm8k():
    """Convert GSM8K parquet to SFT JSONL."""
    out_dir = DATA_DST / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src = GSM8K_SRC / "main" / f"{split}-00000-of-00001.parquet"
        rows = _read_parquet(src)
        print(f"GSM8K {split}: {len(rows)} rows from {src}")

        records = []
        for row in rows:
            question = row["question"]
            answer   = row["answer"]
            prompt   = R1_ZERO_TEMPLATE.format(question=question)
            response = _format_gsm8k_response(answer)

            records.append({
                "question": question,
                "answer":   answer,
                "prompt":   prompt,
                "response": response,
            })

        out_path = out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  → {out_path}  ({len(records)} records)")

        # Also write JSONL for training scripts that expect line-delimited JSON
        jsonl_path = out_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  → {jsonl_path}")


# ──────────────────────────────────────────────
# 2. TULU-3 SFT Personas Math
# ──────────────────────────────────────────────

def convert_tulu():
    """Convert TULU-3 Personas Math parquet to SFT JSONL."""
    out_dir = DATA_DST / "tulu-3-sft-personas-math"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Two shards
    for shard in ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet"):
        src = TULU_SRC / "data" / shard
        rows = _read_parquet(src)
        print(f"TULU {shard}: {len(rows)} rows from {src}")

        for row in rows:
            messages = row["messages"]
            # messages is a list of [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            prompt_text = ""
            response_text = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt_text = msg["content"]
                elif msg["role"] == "assistant":
                    response_text = msg["content"]

            if not prompt_text or not response_text:
                continue

            all_records.append({
                "question": row.get("prompt", prompt_text[:200]),
                "answer":   "",
                "prompt":   prompt_text,
                "response": response_text,
            })

    # Shuffle once for good measure
    import random
    random.shuffle(all_records)
    print(f"TULU total: {len(all_records)} records")

    out_path = out_dir / "train.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path}")

    jsonl_path = out_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → {jsonl_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SFT Data Preparation")
    print("=" * 60)

    print("\n── GSM8K ──")
    convert_gsm8k()

    print("\n── TULU-3 SFT Personas Math ──")
    convert_tulu()

    print("\n✅ Done. All data →", DATA_DST)


if __name__ == "__main__":
    main()
