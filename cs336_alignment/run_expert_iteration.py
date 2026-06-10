"""
run_expert_iteration.py

Entry point for A5 Part 2: Expert Iteration.

Loop: generate → filter correct → SFT → repeat.

Usage:
    uv run python -m cs336_alignment.run_expert_iteration \
        --model_id Qwen/Qwen2.5-Math-1.5B \
        --device cuda:0 --vllm_device cuda:1 \
        --ei_rounds 5 --ei_generations_per_prompt 8
"""
