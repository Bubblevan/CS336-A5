"""
run_dpo.py

Entry point for supplement: DPO on HH-RLHF.

Usage:
    uv run python -m cs336_alignment.run_dpo \
        --model_id meta-llama/Llama-3.2-1B \
        --device cuda:0 --beta 0.1 --max_steps 300
"""
