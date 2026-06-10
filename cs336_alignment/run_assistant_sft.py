"""
run_assistant_sft.py

Entry point for supplement: SFT on UltraChat / safety data.

Usage:
    uv run python -m cs336_alignment.run_assistant_sft \
        --model_id meta-llama/Llama-3.2-1B \
        --device cuda:0 \
        --dataset ultrachat --max_steps 500
"""
