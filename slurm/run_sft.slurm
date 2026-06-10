#!/bin/bash
#SBATCH --job-name=cs336-sft
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

# Load environment
source .venv/bin/activate

# Model & data paths
MODEL_ID="Qwen/Qwen2.5-Math-1.5B"
DATA_DIR="data/datasets/sft-reason"
OUTPUT_DIR="outputs"

# SFT training
uv run python -m cs336_alignment.run_sft \
    --model_id "$MODEL_ID" \
    --device "cuda:0" \
    --vllm_device "cuda:1" \
    --max_steps 200 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --eval_every 50 \
    --lr 1e-5 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "cs336_alignment" \
    --eval_examples 100

echo "SFT training finished."
