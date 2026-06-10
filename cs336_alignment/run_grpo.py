"""
run_grpo.py

Entry point for A5 Part 3: GRPO / GSPO.

Usage:
    uv run python -m cs336_alignment.run_grpo \
        --model_id Qwen/Qwen2.5-Math-1.5B \
        --device cuda:0 --vllm_device cuda:1 \
        --group_size 8 --max_steps 100
"""
