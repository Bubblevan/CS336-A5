# run_reasoning_sft.py
# -----------------------------------------------------------------------------
# CS336 Assignment5 数学推理监督微调（SFT）主入口脚本
# 完整训练流水线闭环：
#   参数解析 → 模型/分词器加载 → Packed SFT数据集构建 → 优化器/调度器初始化 →
#   梯度累积训练循环 → 定期验证评估 → 检查点保存 → 最终模型导出
# 整合了 core 与 reasoning 模块的所有核心能力，是SFT训练的顶层执行逻辑
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import wandb

# 导入自研核心模块：批处理、得分计算、工具函数
from cs336_alignment.core.batching import get_packed_sft_dataset, iterate_batches, collate_fn, split_microbatches
from cs336_alignment.core.scoring import compute_log_probs_from_logits
from cs336_alignment.core.utils import set_seed, write_json
# 导入评估与奖励模块
from cs336_alignment.eval.math import load_math_examples, make_math_prompt, extract_math_answer
from cs336_alignment.reasoning.rewards import r1_zero_reward_fn
# 导入SFT单步损失计算
from cs336_alignment.reasoning.sft import sft_microbatch_train_step


def parse_args() -> argparse.Namespace:
    """
    命令行参数解析：统一管理所有训练超参与路径配置
    按功能分为模型、数据、训练、评估日志四大类，便于维护与调参
    """
    parser = argparse.ArgumentParser(description="SFT on math reasoning traces")

    # ========== 模型相关参数 ==========
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF模型ID或本地检查点路径，作为SFT的基座模型")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练使用的设备，单卡训练默认为 cuda:0")
    parser.add_argument("--vllm_device", type=str, default=None,
                        help="vLLM高速评估使用的设备（如cuda:1），为None则跳过vLLM评估，使用原生HF生成")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="训练精度：bfloat16适合现代Ampere及以上架构，兼顾显存与精度；float16需配合梯度缩放")

    # ========== 数据相关参数 ==========
    parser.add_argument("--data_path", type=str, required=True,
                        help="SFT训练集路径，JSONL格式，每行包含 prompt + response 字段")
    parser.add_argument("--val_path", type=str,
                        default="/root/gpufree-share/data/MATH/validation.jsonl",
                        help="MATH验证集路径，用于训练过程中的生成式评估")
    parser.add_argument("--seq_length", type=int, default=1024,
                        help="Packed SFT数据集的固定序列长度，需适配显存与模型上下文窗口")

    # ========== 训练超参数 ==========
    parser.add_argument("--max_steps", type=int, default=500,
                        help="总训练步数，以优化器更新次数计数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="单卡单微批次的样本数；L40显卡开启flash-attn可设4，关闭则设2")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="梯度累积步数；等效批次大小 = batch_size * grad_accum_steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="峰值学习率；SFT通常使用比预训练小1-2个数量级的学习率")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="学习率预热步数；训练初期线性提升学习率，避免初始大梯度破坏预训练权重")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪最大范数；防止梯度爆炸，稳定大模型训练")

    # ========== 评估与日志参数 ==========
    parser.add_argument("--eval_every", type=int, default=100,
                        help="每N步执行一次验证集评估")
    parser.add_argument("--eval_limit", type=int, default=200,
                        help="评估使用的验证集样本数；0表示跳过评估")
    parser.add_argument("--save_every", type=int, default=100,
                        help="每N步保存一次检查点")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_reasoning",
                        help="输出目录，存放检查点、评估结果、损失日志")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子，保证实验可复现")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训的检查点路径，加载模型、优化器、调度器状态")
    parser.add_argument("--wandb_project", type=str, default="cs336-sft-reasoning",
                        help="Weights & Biases 项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B 运行名称；不填则自动根据超参生成")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B 团队/实体名称")
    parser.add_argument("--patience", type=int, default=3,
                        help="早停耐心值：eval accuracy 连续 N 次不提升则停止训练")
    parser.add_argument("--cooldown_steps", type=int, default=0,
                        help="早停冷却步数：前 N 步不触发早停检查，给 warmup 留空间")

    return parser.parse_args()


def main() -> None:
    """SFT训练主函数：完整的训练生命周期管理"""
    args = parse_args()
    # 固定全局随机种子（Python随机、numpy、torch），保证实验可复现
    set_seed(args.seed)

    # 输出目录：weights 放共享存储（gpufree-share），轻量日志放项目目录
    run_name = args.wandb_run_name or f"sft-{Path(args.model_id).name}-bs{args.batch_size}x{args.grad_accum_steps}-lr{args.lr}"
    weights_dir = Path("/root/gpufree-share/data/sft/checkpoints") / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 从 output_dir 创建软链接指向 weights_dir
    link_path = output_dir / "checkpoints"
    if not link_path.exists():
        rel = Path("../../../gpufree-share/data/sft/checkpoints") / run_name
        link_path.symlink_to(rel)

    # ── 初始化 Weights & Biases 实验追踪 ──
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        # 记录全部超参配置，方便后续实验对比
        config={
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "seq_length": args.seq_length,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
            "max_grad_norm": args.max_grad_norm,
            "eval_limit": args.eval_limit,
            "dtype": args.dtype,
            "data_path": args.data_path,
            "val_path": args.val_path,
        },
    )

    # 设备与精度映射
    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ── 1. 加载分词器与基座模型 ──
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # 兼容处理：很多因果LM（如Llama系列）默认没有pad_token
    # 训练与评估需要padding，因此用eos_token替代pad_token，是行业通用方案
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 自动检测 Flash Attention 可用性
    # Flash Attention 是大模型训练的核心加速组件，可大幅降低显存占用、提升训练速度
    try:
        import flash_attn  # noqa: F401
        _flash_attn_available = True
    except ImportError:
        _flash_attn_available = False

    # 注意力实现选择：有GPU且安装了flash-attn则使用flash_attention_2，否则回退到原生eager实现
    attn_impl = "flash_attention_2" if (torch.cuda.is_available() and _flash_attn_available) else "eager"
    if torch.cuda.is_available() and not _flash_attn_available:
        print("WARNING: flash-attn not installed, falling back to 'eager' attention")

    # 加载因果语言模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,  # 指定加载精度，降低显存占用
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.train()  # 切换为训练模式，启用dropout、batch norm等训练专属层

    # ── 2. 构建 Packed SFT 训练数据集 ──
    print(f"Loading SFT data: {args.data_path}")
    dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=args.data_path,
        seq_length=args.seq_length,
        shuffle=True,  # 打包前打乱文档顺序，避免同主题数据集中
    )
    print(f"  Dataset size: {len(dataset)} sequences")

    # ── 3. 初始化优化器与学习率调度器 ──
    # AdamW：大模型微调的标准优化器，在Adam基础上加入权重衰减正则化，防止过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.max_steps
    # 学习率调度：线性预热 + 线性衰减
    # 预热阶段学习率从0线性升到峰值，之后线性降到0，是SFT最常用的稳定调度策略
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps),  # 预热步数不超过总步数
        num_training_steps=total_steps,
    )

    # ── 4. 断点续训加载 ──
    start_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        # 加载检查点，映射到当前训练设备
        state = torch.load(args.resume, map_location=device)
        # 恢复模型权重、优化器状态、调度器状态
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        # 恢复步数计数，从下一步继续训练
        start_step = state["step"] + 1

    # ── 5. 加载验证集数据 ──
    val_prompts = []
    val_golds = []
    if args.eval_limit > 0:
        # 加载MATH验证集，限制样本数量
        val_examples = load_math_examples(args.val_path, limit=args.eval_limit)
        # 将原始题目格式化为模型要求的prompt格式（加入系统提示、思维链引导等）
        val_prompts = [make_math_prompt(ex["problem"]) for ex in val_examples]
        # 提取标准答案
        val_golds = [ex["answer_raw"] for ex in val_examples]
        print(f"  Validation set: {len(val_prompts)} examples")

    # ── 6. 主训练循环 ──
    print(f"\nStarting training: {args.max_steps} steps, batch_size={args.batch_size}, "
          f"grad_accum={args.grad_accum_steps}, lr={args.lr}")

    global_step = start_step
    # 数据迭代器：一次取出「等效批次大小」的样本
    # 为什么一次取 batch_size * grad_accum_steps 个？
    # → 这些样本会被拆成 grad_accum_steps 个微批次，对应一次完整的优化器更新
    dataloader = iterate_batches(dataset, batch_size=args.batch_size * args.grad_accum_steps, shuffle=True)

    # 日志记录变量
    loss_history = []
    start_time = time.perf_counter()
    step_start_time = time.perf_counter()

    # 早停与最佳检查点追踪
    best_eval_accuracy = 0.0
    best_ckpt_path = weights_dir / "best.pt"
    early_stop_counter = 0
    early_stopped = False

    # 外层循环：控制总训练步数
    while global_step < args.max_steps and not early_stopped:
        # 遍历数据集，逐个取出大批次
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            # 步骤A：将大批次拆分为多个微批次，用于梯度累积
            microbatches = split_microbatches(batch, args.grad_accum_steps)
            accumulated_loss = 0.0  # 记录当前大批次的累计损失，仅用于日志

            # 步骤B：逐个微批次执行前向+反向，梯度累加
            for micro_batch in microbatches:
                # 将数据移到训练设备
                input_ids = micro_batch["input_ids"].to(device)
                labels = micro_batch["labels"].to(device)

                # 模型前向传播，得到logits，形状 (B, L, V)
                logits = model(input_ids).logits

                # 提取每个位置真实标签的对数概率，形状 (B, L)
                log_probs = compute_log_probs_from_logits(logits, labels)

                # ⚠️ Packed SFT 数据集的掩码处理
                # Packed范式下，序列是由多段文档拼接切分而来，几乎没有padding，所有位置都是有效文本
                # 因此这里直接用全1掩码，所有token都参与损失计算
                # 若需严格区分prompt/response，需在数据集阶段记录边界并传入对应掩码
                response_mask = torch.ones_like(log_probs)

                # 计算SFT损失，返回已按梯度累积步数缩放的损失（用于反向）和原始损失（用于日志）
                loss, meta = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=len(microbatches),
                )
                # 反向传播：计算梯度并累加到模型参数的grad属性中
                loss.backward()
                # 累计原始损失，用于日志统计
                accumulated_loss += meta["loss"].item()

            # 步骤C：所有微批次跑完，执行一次优化器更新
            # 梯度裁剪：将参数梯度的范数限制在 max_grad_norm 以内，防止梯度爆炸
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 优化器步进：用累计梯度更新参数
            optimizer.step()
            # 学习率调度器步进：更新学习率
            scheduler.step()
            # 清空梯度，为下一个大批次做准备
            optimizer.zero_grad()

            # 记录损失历史
            loss_history.append(accumulated_loss)

            # 步骤D：日志记录（每步都写入W&B）
            lr_value = scheduler.get_last_lr()[0]
            now = time.perf_counter()
            step_time = now - step_start_time
            step_start_time = now
            tokens_per_step = args.batch_size * args.seq_length * args.grad_accum_steps
            wandb.log({
                "train/loss": accumulated_loss,       # 训练损失
                "train/lr": lr_value,                 # 当前学习率
                "train/tokens_per_sec": tokens_per_step / max(step_time, 1e-8),  # 当前步吞吐率
            }, step=global_step)

            # 控制台输出（每20步打印一次，避免刷屏）
            if global_step % 20 == 0:
                elapsed = time.perf_counter() - start_time
                tok_per_sec = tokens_per_step / max(step_time, 1e-8)
                print(
                    f"  step {global_step:5d}/{args.max_steps} | "
                    f"loss: {accumulated_loss:.4f} | "
                    f"lr: {lr_value:.2e} | "
                    f"tok/s: {tok_per_sec:.0f} | "
                    f"elapsed: {elapsed:.0f}s"
                )

            # 步骤E：定期执行验证评估 + 早停检查
            if args.eval_limit > 0 and global_step % args.eval_every == 0 and global_step > 0:
                eval_metrics = _run_eval(model, tokenizer, val_prompts, val_golds, device, global_step, output_dir, args.eval_limit)
                wandb.log(eval_metrics, step=global_step)

                # 早停判断：冷却步数后才开始检查
                current_acc = eval_metrics.get("eval/accuracy", 0.0)

                # 最佳检查点始终保存（不受 cooldown 影响）
                if current_acc > best_eval_accuracy:
                    best_eval_accuracy = current_acc
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "eval_accuracy": current_acc,
                    }, best_ckpt_path)
                    print(f"  ★ New best accuracy: {current_acc:.4f} → {best_ckpt_path}")

                # 早停计数和触发（受 cooldown 保护）
                if global_step >= args.cooldown_steps:
                    if current_acc <= best_eval_accuracy:
                        early_stop_counter += 1
                        print(f"  Early stop counter: {early_stop_counter}/{args.patience} (best: {best_eval_accuracy:.4f})")
                        if early_stop_counter >= args.patience:
                            print(f"\n  Early stopping triggered at step {global_step} (best accuracy: {best_eval_accuracy:.4f})")
                            early_stopped = True
                            break
                    else:
                        early_stop_counter = 0

            # 步骤F：定期保存检查点（含早停跳出检测）
            if global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = weights_dir / f"checkpoint_{global_step}.pt"
                # 保存完整训练状态，支持断点续训
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": accumulated_loss,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

            global_step += 1

    # ── 7. 训练结束，保存最终模型与日志 ──
    final_path = weights_dir / "final.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    # 保存损失历史
    write_json(output_dir / "loss_history.json", loss_history)

    # W&B收尾
    wandb.log({"train/final_loss": loss_history[-1] if loss_history else 0}, step=global_step)
    wandb.finish()

    elapsed = time.perf_counter() - start_time
    print(f"\nDone! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def _run_eval(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    val_prompts: list[str],
    val_golds: list[str],
    device: torch.device,
    step: int,
    output_dir: Path,
    eval_limit: int,
) -> dict[str, float]:
    """
    在MATH验证集上执行生成式评估，计算准确率与格式合规率
    
    核心注意点（高频易错）：
        1. 评估必须切换模型为eval模式，关闭dropout等训练层
        2. 生成任务必须使用左填充（left padding），右填充会导致生成位置错位
        3. 推理模式下禁用梯度计算，节省显存与耗时
    """
    model.eval()

    # Save original padding side and set to 'left' for generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    correct = 0
    total = 0
    formatted = 0
    results = []

    with torch.inference_mode():
        for i in range(0, len(val_prompts), 8):
            batch_prompts = val_prompts[i : i + 8]
            batch_golds = val_golds[i : i + 8]

            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)

            input_length = encoded["input_ids"].shape[1]
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            new_token_ids = generated_ids[:, input_length:]
            batch_outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

            for prompt, gold, output in zip(batch_prompts, batch_golds, batch_outputs):
                pred = extract_math_answer(output)
                if pred is not None:
                    formatted += 1
                    from cs336_alignment.reasoning.rewards import grade
                    is_correct = grade(pred, gold, fast=True)
                else:
                    is_correct = False

                correct += int(is_correct)
                total += 1
                results.append({
                    "prompt": prompt[:100],
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                })

    accuracy = correct / total if total > 0 else 0.0
    format_rate = formatted / total if total > 0 else 0.0

    metrics = {
        "eval/accuracy": accuracy,
        "eval/format_rate": format_rate,
        "eval/correct": correct,
        "eval/total": total,
    }

    print(f"\n  Eval step {step}: accuracy={accuracy:.4f} ({correct}/{total}), "
          f"format_rate={format_rate:.3f}")

    write_json(output_dir / f"eval_{step}.json", {"step": step, **metrics, "results": results[:20]})

    # Restore original padding side
    tokenizer.padding_side = orig_padding_side

    model.train()
    return metrics


if __name__ == "__main__":
    main()
