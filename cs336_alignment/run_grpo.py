"""
run_grpo.py

GRPO 训练主入口。

流程：
    Step 0:  初始评估（训练前 baseline）
    Loop:
      Phase 1: 从问题池采样 → vLLM 批量生成 G 条/题
      Phase 2: 奖励打分 → 组归一化优势 → 分词 → 旧策略 log-probs
      Phase 3: Inner loop 训练（梯度累积 + 多 epoch）
      Phase 4: 评估 + 保存 checkpoint

用法：
    # Google Cloud HPC（2 卡：cuda:0 HF 训练，cuda:1 vLLM 生成）
    uv run python -m cs336_alignment.run_grpo \
        --model_id Qwen/Qwen2.5-Math-1.5B \
        --train_data data/gsm8k/train.jsonl \
        --val_data data/gsm8k/test.jsonl \
        --prompt_path cs336_alignment/prompts/r1_zero.prompt \
        --device cuda:0 --vllm_device cuda:1 \
        --group_size 8 --rollout_batch_size 128 \
        --train_batch_size 16 --grad_accum_steps 4 \
        --lr 3e-5 --n_grpo_steps 50 \
        --output_dir outputs/grpo

    # 烟雾测试（单卡 HF generate，不用 vLLM）
    uv run python -m cs336_alignment.run_grpo \
        --model_id Qwen/Qwen2.5-Math-1.5B \
        --train_data data/gsm8k/train.jsonl \
        --val_data data/gsm8k/test.jsonl \
        --prompt_path cs336_alignment/prompts/r1_zero.prompt \
        --device cuda:0 --engine hf \
        --group_size 4 --n_grpo_steps 3 \
        --train_limit 20 --eval_limit 20 \
        --output_dir outputs/grpo_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import SamplingParams

from cs336_alignment.core.scoring import get_response_log_probs
from cs336_alignment.core.tokenization import tokenize_prompt_and_output
from cs336_alignment.core.utils import set_seed
from cs336_alignment.core.vllm_utils import VLLMServer
from cs336_alignment.reasoning.grpo_advantage import compute_group_normalized_rewards
from cs336_alignment.reasoning.grpo_loss import compute_policy_gradient_loss
from cs336_alignment.reasoning.prompts import format_math_prompt, load_r1_zero_prompt_template
from cs336_alignment.reasoning.rewards import r1_zero_reward_fn
from cs336_alignment.reasoning.sft import log_generations


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO Training")

    # ── Model ──
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default=None,
                        help="vLLM 设备（如 cuda:1）。None = 用 HF generate 替代 vLLM")
    parser.add_argument("--engine", type=str, default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.3)

    # ── Data ──
    parser.add_argument("--train_data", type=str, required=True,
                        help="训练集 jsonl，每行含 question 和 answer（#### 格式）")
    parser.add_argument("--val_data", type=str, required=True,
                        help="验证集 jsonl，格式同训练集")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="prompt 模板文件路径（如 r1_zero.prompt）")
    parser.add_argument("--train_limit", type=int, default=None,
                        help="限制训练集样本数（调试用）")
    parser.add_argument("--eval_limit", type=int, default=200,
                        help="评估时使用的验证集样本数")

    # ── Rollout ──
    parser.add_argument("--group_size", type=int, default=8,
                        help="每问题生成 G 条回答")
    parser.add_argument("--rollout_batch_size", type=int, default=128,
                        help="一轮 rollout 的总回答数 = 问题数 × group_size")
    parser.add_argument("--sampling_temperature", type=float, default=0.6)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)

    # ── Training ──
    parser.add_argument("--n_grpo_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="逻辑 batch size（单个 optimizer step 的样本数）")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="梯度累积步数，micro_batch_size = train_batch_size // grad_accum_steps")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1,
                        help="同一批 rollout 数据训练几个 epoch；>1 为 off-policy")
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="grpo_clip",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--normalize_by_std", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--length_norm_type", type=str, default="mask_normalize",
                        choices=["mask_mean", "mask_normalize"])
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="最大序列长度，用于 mask_normalize 的归一化常数和超长过滤")

    # ── KL & Stability ──
    parser.add_argument("--kl_coef", type=float, default=0.0,
                        help="KL 散度惩罚系数，0=禁用。建议 0.01~0.1")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="连续 N 次 eval 不创新高时停止。0=禁用")

    # ── Eval & Save ──
    parser.add_argument("--eval_every_steps", type=int, default=10)
    parser.add_argument("--save_every_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="cs336-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def _extract_gold(answer: str) -> str:
    """从 GSM8K 的 `#### N` 格式中提取数值答案。"""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def load_question_pool(
    data_path: str | Path,
    prompt_template: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """
    从 jsonl 加载问题池。

    支持两种数据格式：
        - MATH 格式：{"problem": "...", "answer": "..."}
        - GSM8K 格式：{"question": "...", "answer": "..."}

    每条返回：
        {"prompt": 格式化后的完整 prompt, "gold": 标准答案字符串}
    """
    pool: list[dict[str, str]] = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            # Support both MATH ("problem") and GSM8K ("question") formats
            question = item.get("problem") or item.get("question", "")
            if not question:
                continue
            gold = _extract_gold(item.get("gold", item.get("answer", "")))
            prompt = prompt_template.replace("{question}", question)
            pool.append({"prompt": prompt, "gold": gold})
    return pool


# ═══════════════════════════════════════════════════════════════════
# Rollout helpers
# ═══════════════════════════════════════════════════════════════════

def flatten_rollout_outputs(
    batch_questions: list[dict[str, str]],
    outputs,
) -> tuple[list[str], list[str], list[str]]:
    """
    将 vLLM 的 group rollout 结果展平。

    返回：
        flat_prompts:    [prompt_i] × G
        flat_responses:  [response_i_1, response_i_2, ..., response_i_G]
        flat_golds:      [gold_i] × G
    """
    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    flat_golds: list[str] = []
    for q_item, out in zip(batch_questions, outputs):
        for candidate in out.outputs:
            flat_prompts.append(q_item["prompt"])
            flat_responses.append(candidate.text)
            flat_golds.append(q_item["gold"])
    return flat_prompts, flat_responses, flat_golds


def precompute_old_log_probs(
    policy: PreTrainedModel,
    tokenized_data: dict[str, Tensor],
    micro_batch_size: int,
    device: str,
) -> Tensor:
    """用当前 policy（old policy）预计算 rollout 数据上的 token log-probs。"""
    policy.eval()
    all_old_log_probs: list[Tensor] = []
    total = tokenized_data["input_ids"].size(0)
    with torch.no_grad():
        for start in range(0, total, micro_batch_size):
            end = min(start + micro_batch_size, total)
            mb = {
                k: v[start:end].to(device)
                for k, v in tokenized_data.items()
            }
            res = get_response_log_probs(policy, mb["input_ids"], mb["labels"])
            all_old_log_probs.append(res["log_probs"].cpu())
    policy.train()
    return torch.cat(all_old_log_probs, dim=0)


# ═══════════════════════════════════════════════════════════════════
# GRPO inner training loop
# ═══════════════════════════════════════════════════════════════════

def train_one_grpo_step(
    policy: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    labels: Tensor,
    response_mask: Tensor,
    advantages: Tensor,
    raw_rewards: Tensor,
    old_log_probs: Tensor,
    *,
    train_batch_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    loss_type: str,
    cliprange: float,
    length_norm_type: str,
    normalize_constant: float | None,
    max_grad_norm: float,
    epochs_per_rollout_batch: int,
    kl_coef: float = 0.0,
    ref_model: PreTrainedModel | None = None,
) -> dict[str, float]:
    """
    对一批 rollout 数据做 inner loop 训练。

    返回聚合后的训练指标（loss / clip_fraction / ratio_mean / grad_norm / kl）。
    """
    total_samples = input_ids.size(0)
    num_updates = total_samples // train_batch_size

    if num_updates == 0:
        return {"loss": 0.0}

    agg_loss = 0.0
    agg_clip = 0.0
    agg_ratio = 0.0
    agg_kl = 0.0
    agg_grad_norm = 0.0
    agg_response_entropy = 0.0
    n_steps = 0

    for _ in range(epochs_per_rollout_batch):
        indices = torch.randperm(total_samples)
        for update_step in range(num_updates):
            batch_start = update_step * train_batch_size
            batch_end = (update_step + 1) * train_batch_size
            batch_idx = indices[batch_start:batch_end]

            optimizer.zero_grad()

            batch_loss = 0.0
            batch_clip = 0.0
            batch_ratio = 0.0
            batch_kl = 0.0
            batch_response_entropy = 0.0

            for micro_step in range(grad_accum_steps):
                m_start = micro_step * micro_batch_size
                m_end = (micro_step + 1) * micro_batch_size
                micro_idx = batch_idx[m_start:m_end]

                mb_input_ids = input_ids[micro_idx].to(policy.device)
                mb_labels = labels[micro_idx].to(policy.device)
                mb_mask = response_mask[micro_idx].to(policy.device)
                mb_adv = advantages[micro_idx].unsqueeze(-1).to(policy.device)
                mb_raw = raw_rewards[micro_idx].unsqueeze(-1).to(policy.device)
                mb_old = old_log_probs[micro_idx].to(policy.device)

                # Forward
                res = get_response_log_probs(policy, mb_input_ids, mb_labels,
                                             return_token_entropy=True)
                policy_log_probs = res["log_probs"]
                token_entropy = res.get("token_entropy")

                # Per-token loss
                per_token_loss, meta = compute_policy_gradient_loss(
                    policy_log_probs=policy_log_probs,
                    loss_type=loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old,
                    cliprange=cliprange,
                )

                # Aggregate
                if length_norm_type == "mask_normalize":
                    assert normalize_constant is not None
                    total = (per_token_loss * mb_mask).sum()
                    microbatch_loss = total / normalize_constant
                    scaled_loss = microbatch_loss
                else:  # mask_mean
                    per_seq = (per_token_loss * mb_mask).sum(dim=1) / mb_mask.sum(dim=1).clamp(min=1)
                    microbatch_loss = per_seq.mean()
                    scaled_loss = microbatch_loss / grad_accum_steps

                scaled_loss.backward()

                # ── KL divergence penalty ──
                if kl_coef > 0 and ref_model is not None:
                    with torch.no_grad():
                        ref_res = get_response_log_probs(ref_model, mb_input_ids, mb_labels)
                        ref_log_probs = ref_res["log_probs"]
                    # Approximate KL: ρ - log(ρ) - 1 where ρ = exp(ref_lp - policy_lp)
                    log_ratio = ref_log_probs - policy_log_probs
                    ratio = torch.exp(log_ratio)
                    kl_per_token = ratio - log_ratio - 1.0
                    # Masked mean over response tokens
                    kl_scalar = (kl_per_token * mb_mask).sum() / mb_mask.sum().clamp(min=1)
                    (kl_coef * kl_scalar / grad_accum_steps).backward()
                else:
                    kl_scalar = torch.tensor(0.0)

                with torch.no_grad():
                    batch_loss += microbatch_loss.detach().item()
                    batch_clip += meta.get("clip_fraction", torch.tensor(0.0)).detach().item()
                    batch_ratio += meta.get("ratio_mean", torch.tensor(1.0)).detach().item()
                    batch_kl += float(kl_scalar.item()) if kl_coef > 0 else 0.0

                    # Entropy on response tokens
                    if token_entropy is not None:
                        res_mask = mb_mask.bool()
                        e = token_entropy[res_mask]
                        batch_response_entropy += e.mean().item() if e.numel() > 0 else 0.0

            # Clip + step
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            agg_loss += batch_loss
            agg_clip += batch_clip
            agg_ratio += batch_ratio
            agg_kl += batch_kl
            agg_grad_norm += gn.item()
            agg_response_entropy += batch_response_entropy
            n_steps += 1

    denom = max(n_steps, 1)
    return {
        "loss": agg_loss / denom,
        "clip_fraction": agg_clip / denom,
        "ratio_mean": agg_ratio / denom,
        "kl_approx": agg_kl / denom,
        "grad_norm": agg_grad_norm / denom,
        "response_entropy": agg_response_entropy / denom,
    }


# ═══════════════════════════════════════════════════════════════════
# HF eval helper（替代 log_generations 的 vLLM-only 接口）
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _hf_eval(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn,
    max_new_tokens: int,
) -> dict[str, float]:
    """用 HF generate 做评估，返回和 log_generations 一致的 metrics 字典。"""
    # Disable gradient checkpointing for generation (can cause shape issues)
    was_checkpointing = policy.is_gradient_checkpointing
    if was_checkpointing:
        policy.gradient_checkpointing_disable()

    total = len(prompts)
    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0
    response_lengths = []

    for prompt, gt in zip(prompts, ground_truths):
        inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)
        # Skip empty prompts
        if inputs.input_ids.shape[1] == 0:
            continue
        out = policy.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        scores = reward_fn(generated, gt)
        r = scores.get("reward", 0.0)
        total_reward += r
        total_format += scores.get("format_reward", 0.0)
        total_answer += scores.get("answer_reward", 0.0)
        response_lengths.append(len(generated))

    # Re-enable gradient checkpointing if it was on
    if was_checkpointing:
        policy.gradient_checkpointing_enable()

    n = max(total, 1)
    return {
        "eval/accuracy": total_reward / n,
        "eval/avg_format_reward": total_format / n,
        "eval/avg_answer_reward": total_answer / n,
        "eval/avg_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
    }


def _run_eval(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    vllm_server: VLLMServer | None,
    val_samples: list[dict[str, str]],
    sampling_params,
    use_vllm: bool,
    max_new_tokens: int,
) -> dict[str, float]:
    """统一评估入口：vLLM 模式走 REST API，HF 模式走 _hf_eval。"""
    if use_vllm and vllm_server is not None:
        vllm_server.sync_policy_weights(policy)
        # Use REST API directly (VLLMServer has generate_completions, not .generate())
        prompts = [s["prompt"] for s in val_samples]
        ground_truths = [s["gold"] for s in val_samples]
        api_params = {
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "n": 1,
        }
        completions = vllm_server.generate_completions(prompts, api_params)
        total_reward = 0.0
        response_lengths = []
        for i, c in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else ""
            scores = r1_zero_reward_fn(c.text, gt)
            r = scores.get("reward", 0.0)
            total_reward += r
            response_lengths.append(len(c.text))
        n = max(len(completions), 1)
        return {
            "eval/accuracy": total_reward / n,
            "eval/avg_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
        }
    else:
        return _hf_eval(
            policy=policy,
            tokenizer=tokenizer,
            prompts=[s["prompt"] for s in val_samples],
            ground_truths=[s["gold"] for s in val_samples],
            reward_fn=r1_zero_reward_fn,
            max_new_tokens=max_new_tokens,
        )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_grpo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    assert args.train_batch_size % args.grad_accum_steps == 0
    micro_batch_size = args.train_batch_size // args.grad_accum_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size

    # ── Wandb ──
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"grpo_lr{args.lr}_g{args.group_size}",
        config=vars(args),
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("global_step")
    wandb.define_metric("rollout/*", step_metric="grpo_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    # ── Output dir & shared weights storage ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or f"grpo_lr{args.lr}_g{args.group_size}"
    weights_dir = Path("/root/gpufree-share/data/sft/checkpoints") / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)

    link_path = output_dir / "checkpoints"
    link_path.unlink(missing_ok=True)
    link_path.symlink_to(Path("../../../gpufree-share/data/sft/checkpoints") / run_name)

    # ── Prompt template & data ──
    print(f"Loading prompt template: {args.prompt_path}")
    prompt_template = load_r1_zero_prompt_template(args.prompt_path)

    print(f"Loading training data: {args.train_data}")
    question_pool = load_question_pool(args.train_data, prompt_template, limit=args.train_limit)
    print(f"  → {len(question_pool)} training questions")

    print(f"Loading validation data: {args.val_data}")
    val_samples = load_question_pool(args.val_data, prompt_template, limit=args.eval_limit)
    print(f"  → {len(val_samples)} validation questions")

    # ── Model ──
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Try flash_attention_2, fall back to eager if not installed
    attn_impl = "flash_attention_2"
    try:
        policy = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    except ImportError:
        print("FlashAttention2 not available, falling back to eager attention.")
        attn_impl = "eager"
        policy = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    policy = policy.to(args.device)
    policy.gradient_checkpointing_enable()
    policy.train()

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    # ── Reference model for KL divergence ──
    ref_model: PreTrainedModel | None = None
    if args.kl_coef > 0:
        print(f"Loading reference model for KL divergence (kl_coef={args.kl_coef})...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        ).to(args.device)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        print("Reference model ready (frozen).")

    # ── vLLM ──
    use_vllm = args.engine == "vllm" and args.vllm_device is not None
    vllm_server: VLLMServer | None = None
    if use_vllm:
        print(f"Starting vLLM server on {args.vllm_device}...")
        vllm_server = VLLMServer(
            model_id=args.model_id,
            gpu=int(args.vllm_device.split(":")[-1]),
            gpu_memory_utilization=args.vllm_gpu_util,
            seed=args.seed,
        )
        vllm_server.start()
        vllm_server.init_weight_sync(args.device)
        print("vLLM server ready.")

    # ── Sampling params ──
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    rollout_sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        # IMPORTANT: do NOT pass seed. vLLM with n>1 + fixed seed
        # will produce identical responses for all n candidates,
        # killing GRPO's group diversity.  Leave seed unset so
        # each of the n candidates gets independent sampling.
    )

    # ── Step 0: Initial evaluation ──
    print("\n[Step 0] Initial evaluation...")
    policy.eval()
    if use_vllm and vllm_server is not None:
        vllm_server.sync_policy_weights(policy)

    metrics = _run_eval(
        policy=policy,
        tokenizer=tokenizer,
        vllm_server=vllm_server,
        val_samples=val_samples,
        sampling_params=eval_sampling_params,
        use_vllm=use_vllm,
        max_new_tokens=args.sampling_max_tokens,
    )
    print(f"  Initial accuracy: {metrics.get('eval/accuracy', 0):.2%}")
    wandb.log({"eval/accuracy": metrics.get("eval/accuracy", 0), "global_step": 0})
    policy.train()

    # ── GRPO loop ──
    global_step = 0
    best_eval_acc = 0.0
    eval_no_improve_count = 0
    progress = tqdm(range(1, args.n_grpo_steps + 1), desc="GRPO")

    for grpo_step in progress:
        step_start = time.time()

        # ═══════════════════════════════════════════
        # Phase 1: Rollout
        # ═══════════════════════════════════════════
        policy.eval()
        if vllm_server is not None:
            vllm_server.sync_policy_weights(policy)

        batch_questions = random.sample(question_pool, min(n_prompts_per_rollout, len(question_pool)))
        prompts = [q["prompt"] for q in batch_questions]
        golds = [q["gold"] for q in batch_questions]

        if use_vllm:
            assert vllm_server is not None
            # Convert SamplingParams → dict for the REST API.
            # Explicitly omit seed so each of the n candidates
            # within a group gets independent sampling.
            rollout_dict = {
                "temperature": rollout_sampling_params.temperature,
                "max_tokens": rollout_sampling_params.max_tokens,
                "min_tokens": rollout_sampling_params.min_tokens,
                "n": rollout_sampling_params.n,
                "stop": rollout_sampling_params.stop,
                "include_stop_str_in_output": rollout_sampling_params.include_stop_str_in_output,
            }
            outputs = vllm_server.generate_completions(
                prompts=prompts,
                sampling_params=rollout_dict,
            )
            # VLLMServer returns flat list (prompts × group_size)
            # Re-group for flatten_rollout_outputs
            class MockOutput:
                def __init__(self, outputs_slice):
                    self.outputs = outputs_slice
            regrouped = []
            for i in range(len(batch_questions)):
                start = i * args.group_size
                end = (i + 1) * args.group_size
                regrouped.append(MockOutput(outputs[start:end]))

            flat_prompts, flat_responses, flat_golds = flatten_rollout_outputs(
                batch_questions, regrouped,
            )
        else:
            # HF generate fallback (no group sampling in one call)
            flat_prompts = []
            flat_responses = []
            flat_golds = []
            for q_item in batch_questions:
                for _ in range(args.group_size):
                    # Naive: generate one response at a time via HuggingFace
                    inputs = tokenizer(q_item["prompt"], return_tensors="pt").to(args.device)
                    out = policy.generate(**inputs, max_new_tokens=args.sampling_max_tokens,
                                          temperature=args.sampling_temperature,
                                          do_sample=args.sampling_temperature > 0)
                    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    flat_prompts.append(q_item["prompt"])
                    flat_responses.append(response)
                    flat_golds.append(q_item["gold"])

        # ═══════════════════════════════════════════
        # Phase 2: Rewards → Advantages → Old log-probs
        # ═══════════════════════════════════════════
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_golds,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )

        # 计算平均响应长度和多样性指标
        rollout_lengths = [len(r) for r in flat_responses]
        avg_rollout_length = np.mean(rollout_lengths) if rollout_lengths else 0.0

        # 组内唯一率：每组 unique response 占比，衡量 rollout 多样性
        grouped = [flat_responses[i:i+args.group_size]
                   for i in range(0, len(flat_responses), args.group_size)]
        unique_ratios = [len(set(g)) / len(g) for g in grouped]
        avg_unique_ratio = np.mean(unique_ratios) if unique_ratios else 0.0

        wandb.log({
            "rollout/mean_reward": reward_meta["mean_reward"],
            "rollout/reward_std": reward_meta["std_reward"],
            "rollout/max_reward": reward_meta["max_reward"],
            "rollout/mean_advantage": reward_meta["mean_advantage"],
            "rollout/num_responses": len(flat_responses),
            "rollout/avg_response_length": avg_rollout_length,
            "rollout/unique_ratio": avg_unique_ratio,
            "grpo_step": grpo_step,
        })

        # Tokenize
        tokenized = tokenize_prompt_and_output(flat_prompts, flat_responses, tokenizer)
        input_ids = tokenized["input_ids"]   # (B, L-1)
        labels = tokenized["labels"]         # (B, L-1)
        response_mask = tokenized["response_mask"]  # (B, L-1)

        B = input_ids.size(0)

        # Length filtering: skip samples exceeding max_seq_len
        valid_mask = (response_mask.sum(dim=1) <= args.max_seq_len)
        if not valid_mask.any():
            print("⚠️ All samples exceed max_seq_len, skipping step.")
            policy.train()
            continue

        input_ids = input_ids[valid_mask]
        labels = labels[valid_mask]
        response_mask = response_mask[valid_mask]
        advantages = advantages[valid_mask]
        raw_rewards = raw_rewards[valid_mask]

        # Resample if too few
        actual_b = input_ids.size(0)
        if actual_b < args.train_batch_size:
            repeat_times = math.ceil(args.train_batch_size / actual_b)
            idx = torch.arange(actual_b).repeat(repeat_times)[:args.train_batch_size]
            input_ids = input_ids[idx]
            labels = labels[idx]
            response_mask = response_mask[idx]
            advantages = advantages[idx]
            raw_rewards = raw_rewards[idx]

        # Precompute old log_probs
        tokenized_cpu = {
            "input_ids": input_ids.cpu(),
            "labels": labels.cpu(),
            "response_mask": response_mask.cpu(),
        }
        old_log_probs = precompute_old_log_probs(
            policy, tokenized_cpu, micro_batch_size, args.device,
        )

        # Normalize constant for mask_normalize
        normalize_constant = None
        if args.length_norm_type == "mask_normalize":
            normalize_constant = float(args.max_seq_len)

        # ═══════════════════════════════════════════
        # Phase 3: Inner loop training
        # ═══════════════════════════════════════════
        policy.train()
        train_metrics = train_one_grpo_step(
            policy=policy,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            response_mask=response_mask,
            advantages=advantages,
            raw_rewards=raw_rewards,
            old_log_probs=old_log_probs,
            train_batch_size=args.train_batch_size,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            loss_type=args.loss_type,
            cliprange=args.cliprange,
            length_norm_type=args.length_norm_type,
            normalize_constant=normalize_constant,
            max_grad_norm=args.max_grad_norm,
            epochs_per_rollout_batch=args.epochs_per_rollout_batch,
            kl_coef=args.kl_coef,
            ref_model=ref_model,
        )

        global_step += 1
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "train/loss": train_metrics["loss"],
            "train/clip_fraction": train_metrics["clip_fraction"],
            "train/ratio_mean": train_metrics["ratio_mean"],
            "train/kl_approx": train_metrics["kl_approx"],
            "train/grad_norm": train_metrics["grad_norm"],
            "train/response_entropy": train_metrics["response_entropy"],
            "train/lr": current_lr,
            "global_step": global_step,
        })

        step_time = time.time() - step_start
        progress.set_postfix({
            "loss": f"{train_metrics['loss']:.4f}",
            "clip": f"{train_metrics['clip_fraction']:.3f}",
            "time": f"{step_time:.1f}s",
        })

        # ═══════════════════════════════════════════
        # Phase 4: Evaluation & checkpoint
        # ═══════════════════════════════════════════
        if grpo_step % args.eval_every_steps == 0:
            policy.eval()
            metrics = _run_eval(
                policy=policy,
                tokenizer=tokenizer,
                vllm_server=vllm_server,
                val_samples=val_samples,
                sampling_params=eval_sampling_params,
                use_vllm=use_vllm,
                max_new_tokens=args.sampling_max_tokens,
            )
            print(f"\n[Step {grpo_step}] Eval accuracy: {metrics.get('eval/accuracy', 0):.2%}")
            wandb.log({
                "eval/accuracy": metrics.get("eval/accuracy", 0),
                "eval/avg_reward": metrics.get("eval/avg_reward", 0),
                "eval/avg_response_length": metrics.get("eval/avg_response_length", 0),
                "global_step": global_step,
            })
            policy.train()

        # ── Early stopping ──
        if args.early_stopping_patience > 0 and grpo_step > 0:
            eval_acc = metrics.get("eval/accuracy", 0.0) if grpo_step % args.eval_every_steps == 0 else None
            if eval_acc is not None:
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    eval_no_improve_count = 0
                else:
                    eval_no_improve_count += 1
                    if eval_no_improve_count >= args.early_stopping_patience:
                        print(f"\n🛑 Early stopping at step {grpo_step}: eval accuracy {eval_acc:.1%} "
                              f"did not improve for {args.early_stopping_patience} consecutive evals "
                              f"(best: {best_eval_acc:.1%})")
                        break

        if grpo_step % args.save_every_steps == 0:
            save_path = weights_dir / f"grpo_step{grpo_step}"
            save_path.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Checkpoint saved: {save_path} (symlink: {link_path})")

        torch.cuda.empty_cache()

    # ── Cleanup ──
    if vllm_server is not None:
        vllm_server.stop()
    wandb.finish()
    print("\nGRPO training finished.")


def main() -> None:
    args = parse_args()
    run_grpo(args)


if __name__ == "__main__":
    main()
