#!/bin/bash
#SBATCH --job-name=cs336-grpo
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Load environment
source .venv/bin/activate

# Model & data paths
MODEL_ID="Qwen/Qwen2.5-Math-1.5B"
DATA_DIR="data/datasets/sft-reason"
OUTPUT_DIR="outputs"

# GRPO / Expert Iteration training
uv run python -m cs336_alignment.run_grpo \
    --model_id "$MODEL_ID" \
    --device "cuda:0" \
    --vllm_device "cuda:1" \
    --group_size 8 \
    --max_steps 100 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --lr 1e-6 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "cs336_alignment" \
    --eval_examples 100

echo "GRPO training finished."
