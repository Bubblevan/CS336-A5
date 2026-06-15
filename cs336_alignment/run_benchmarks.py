"""Unified benchmark runner.

V1 更新：
- V0 只支持 GSM8K；
- 新增 --engine vllm/hf；
- vLLM 路径用于快速 offline batched inference；
- HF 路径保留，用于 fallback 和调试 Flash Attention。

Example vLLM:
    CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_alignment.run_benchmarks \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --engine vllm \
        --benchmarks gsm8k \
        --gsm8k_path data/gsm8k/main/test-00000-of-00001.parquet \
        --output_dir outputs/baseline_qwen_math_vllm \
        --max_new_tokens 512

Example HF fallback:
    uv run python -m cs336_alignment.run_benchmarks \
        --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
        --engine hf \
        --benchmarks gsm8k \
        --gsm8k_path data/gsm8k/main/test-00000-of-00001.parquet \
        --output_dir outputs/baseline_qwen_math_hf \
        --device cuda:0 \
        --hf_batch_size 8 \
        --attn_implementation flash_attention_2 \
        --max_new_tokens 512
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cs336_alignment.core.utils import (
    get_model_and_tokenizer,
    set_seed,
    write_json,
)
from cs336_alignment.eval.generation import (
    GenerationConfig as HFGenerator,
    VLLMGenerationConfig as VLLMGenerator,
)
from cs336_alignment.eval.gsm8k import run_gsm8k_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "hf"],
        default="vllm",
        help="Inference backend. Use vllm for fast offline benchmark eval.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k",
        help="Comma-separated benchmark names. V1 supports: gsm8k.",
    )
    parser.add_argument(
        "--gsm8k_path",
        type=str,
        default="data/gsm8k/main/test-00000-of-00001.parquet",
        help="Path to GSM8K parquet/jsonl file or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline",
        help="Directory for summary.json and predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "HF backend uses this directly. "
            "For vLLM, prefer setting CUDA_VISIBLE_DEVICES before launching; "
            "this script will also set it from cuda:N if absent."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N examples. Use for smoke tests.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Generation budget per example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Shared-ish generation options.
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=None,
        help=(
            "Optional stop strings. Example: --stop_strings '</answer>' "
            "Usually leave empty for baseline to avoid truncating useful reasoning."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # HF backend options.
    parser.add_argument(
        "--hf_batch_size",
        type=int,
        default=8,
        help="Batch size for HF generate fallback.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help=(
            "HF attention backend. Use flash_attention_2 if installed; "
            "use eager to debug compatibility."
        ),
    )

    # vLLM backend options.
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help=(
            "Optional vLLM max model length. "
            "Set e.g. 2048 or 4096 to reduce KV cache memory if needed."
        ),
    )
    parser.add_argument(
        "--vllm_chunk_size",
        type=int,
        default=512,
        help="Number of prompts sent to vLLM per chunk.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM prefix caching. GSM8K prompts share a long prefix.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Debug option for vLLM. Usually keep false.",
    )

    return parser.parse_args()


def maybe_set_cuda_visible_devices_for_vllm(device: str) -> None:
    """Set CUDA_VISIBLE_DEVICES for vLLM if the user passed --device cuda:N.

    vLLM generally chooses visible GPUs rather than a torch-style device string.
    For single-GPU experiments, this helper makes --device cuda:0 behave naturally.

    If CUDA_VISIBLE_DEVICES is already set by the shell, we respect it.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    if not device.startswith("cuda:"):
        return

    gpu_id = device.split(":", maxsplit=1)[1]
    if gpu_id.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def build_generator(args: argparse.Namespace):
    """Construct selected inference backend."""
    if args.engine == "vllm":
        maybe_set_cuda_visible_devices_for_vllm(args.device)

        return VLLMGenerator(
            model_id_or_dir=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_length=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            chunk_size=args.vllm_chunk_size,
        )

    if args.engine == "hf":
        model, tokenizer = get_model_and_tokenizer(
            args.model_id,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )

        return HFGenerator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.hf_batch_size,
        )

    raise ValueError(f"Unsupported engine: {args.engine}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_names = {
        name.strip().lower()
        for name in args.benchmarks.split(",")
        if name.strip()
    }

    unsupported = benchmark_names - {"gsm8k"}
    if unsupported:
        raise ValueError(
            f"Unsupported benchmarks in V1: {sorted(unsupported)}. "
            "Currently only 'gsm8k' is implemented."
        )

    print(f"Building inference backend: {args.engine}")
    print(f"Model: {args.model_id}")
    generator = build_generator(args)

    summaries = {
        "config": {
            "model_id": args.model_id,
            "engine": args.engine,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "seed": args.seed,
        }
    }

    if "gsm8k" in benchmark_names:
        predictions_path = output_dir / "gsm8k_predictions.jsonl"

        print(f"Running GSM8K eval on: {args.gsm8k_path}")
        gsm8k_summary = run_gsm8k_eval(
            generator=generator,
            gsm8k_data_path=args.gsm8k_path,
            split="test",
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            output_path=predictions_path,
            stop_strings=args.stop_strings,
        )

        summaries["gsm8k"] = gsm8k_summary

        print("\nGSM8K summary:")
        for key, value in gsm8k_summary.items():
            print(f"  {key}: {value}")

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summaries)

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()