"""
run_expert_iteration.py

Expert Iteration 入口：生成 → 筛选 → SFT → 重复多轮。

Usage:
    初始模型从 SFT checkpoint 启动：
    uv run python -m cs336_alignment.run_expert_iteration \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --device cuda:0 \
        --data_path /root/gpufree-share/data/MATH/train.jsonl \
        --val_path /root/gpufree-share/data/MATH/validation.jsonl \
        --ei_rounds 3 \
        --ei_generations_per_prompt 4 \
        --ei_sft_epochs 1 \
        --ei_batch_size 4 \
        --ei_temperature 0.7 \
        --eval_limit 200 \
        --output_dir outputs/expert_iteration_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_alignment.reasoning.ei import expert_iteration_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expert Iteration")

    # Model
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF模型ID或本地检查点路径。从SFT checkpoint启动。")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备，单卡默认为cuda:0")
    parser.add_argument("--vllm_device", type=str, default=None,
                        help="vLLM生成设备（如cuda:1）。设置后 rollout 用 vLLM 生成，SFT 用 HF 训练")
    parser.add_argument("--engine", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Rollout生成引擎。hf=HF generate（慢）, vllm=vLLM（需双卡）")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    # Data
    parser.add_argument("--data_path", type=str,
                        default="/root/gpufree-share/data/MATH/train.jsonl",
                        help="MATH训练集路径，用于生成rollout和SFT")
    parser.add_argument("--val_path", type=str,
                        default="/root/gpufree-share/data/MATH/validation.jsonl",
                        help="MATH验证集路径，用于每轮评估")
    parser.add_argument("--ei_train_limit", type=int, default=None,
                        help="限制训练样本数（调试用）")

    # EI params
    parser.add_argument("--ei_rounds", type=int, default=3,
                        help="Expert Iteration轮数")
    parser.add_argument("--ei_generations_per_prompt", type=int, default=4,
                        help="每个问题生成的rollout数量")
    parser.add_argument("--ei_sft_epochs", type=int, default=1,
                        help="每轮SFT训练的epoch数")
    parser.add_argument("--ei_batch_size", type=int, default=4,
                        help="生成/SFT的batch size（L40 4GB+）")
    parser.add_argument("--ei_max_new_tokens", type=int, default=512,
                        help="每段rollout的最大生成token数")
    parser.add_argument("--ei_temperature", type=float, default=0.0,
                        help="生成rollout的采样温度；0.0=贪心(推荐首轮)，>0增加多样性")

    # Eval
    parser.add_argument("--eval_limit", type=int, default=200,
                        help="每轮评估的验证集样本数")
    parser.add_argument("--output_dir", type=str, default="outputs/expert_iteration",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert_iteration_loop(args)


if __name__ == "__main__":
    main()
