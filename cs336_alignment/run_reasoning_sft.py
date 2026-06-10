"""
run_reasoning_sft.py

Entry point for A5 Part 1: SFT on math reasoning traces.

Usage:
    uv run python -m cs336_alignment.run_reasoning_sft \
        --model_id Qwen/Qwen2.5-Math-1.5B \
        --device cuda:0 --vllm_device cuda:1 \
        --max_steps 200 --batch_size 8
"""
