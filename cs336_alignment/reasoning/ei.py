# -----------------------------------------------------------------------------
# Expert Iteration（专家迭代，简称EI）核心实现
# 算法定位：介于监督微调(SFT)与强化学习(RL)之间的推理对齐方法
# 核心思想（自我迭代提升）：
#   1. 用当前策略模型批量生成推理轨迹（rollout）
#   2. 通过奖励函数筛选出正确的推理轨迹，作为「自生成专家数据」
#   3. 用筛选出的正确数据对模型做一轮SFT，更新策略
#   4. 重复多轮迭代，模型的推理能力会逐步自举式提升
# 是 DeepSeek R1、OpenAI o1 等推理模型的核心算法范式，也是CS336 Assignment5的重点内容
# 本模块大量复用前文SFT模块的能力（打包数据集、损失计算、梯度累积），避免重复实现
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import wandb

# 复用SFT核心模块
from cs336_alignment.core.batching import get_packed_sft_dataset, iterate_batches, split_microbatches
from cs336_alignment.core.scoring import compute_log_probs_from_logits, get_response_log_probs
from cs336_alignment.core.utils import set_seed, write_json
# 复用数学评估工具
from cs336_alignment.eval.math import load_math_examples, make_math_prompt, extract_math_answer
from cs336_alignment.reasoning.rewards import r1_zero_reward_fn, grade
# 复用SFT单步损失计算
from cs336_alignment.reasoning.sft import sft_microbatch_train_step


def convert_reasoning_records_to_sft_jsonl(
    source_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    seed: int = 42,
) -> int:
    """
    将EI生成的推理轨迹记录，转换为标准SFT训练所需的 {prompt, response} JSONL格式。

    设计目的：
        EI生成的原始rollout记录包含得分、元信息等额外字段，
        而PackedSFTDataset只识别prompt+response字段，因此需要做格式转换，
        从而可以100%复用SFT的训练流水线，无需重复开发数据加载逻辑。

    Args:
        source_path: EI轮次产出的 rollout_records.jsonl 源文件路径
        output_path: 转换后的SFT格式输出路径
        limit: 可选，仅随机抽取N条记录转换，用于控制单轮训练数据量
        seed: 抽样的随机种子，保证可复现

    Returns:
        实际写入的记录条数
    """
    # 读取所有原始记录
    records = []
    with open(source_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 若设置了limit，则随机抽样指定数量的记录
    if limit is not None and limit < len(records):
        random.seed(seed)
        records = random.sample(records, limit)

    # 逐行写入标准SFT格式
    count = 0
    with open(output_path, "w") as f:
        for rec in records:
            prompt = rec.get("prompt", "")
            response = rec.get("response", "")
            # 仅保留prompt和response都非空的有效样本
            if prompt and response:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                }, ensure_ascii=False) + "\n")
                count += 1

    return count


def expert_iteration_round(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    train_problems: list[dict[str, str]],
    reward_fn: Any,
    *,
    n_generations: int = 4,
    batch_size: int = 4,
    sft_epochs: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    vllm_generator: Any = None,
    wandb_step: list[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    执行单轮专家迭代：生成推理轨迹 → 筛选正确样本 → 监督微调更新模型。

    算法三阶段详解（CS336核心考点）：
        Phase 1 Rollout：每个题目生成n条推理路径，用温度采样保证多样性，
            目的是探索更多解题方式，增加找到正确轨迹的概率。
        Phase 2 Filter：用奖励函数判分，只保留答对的轨迹作为训练数据。
            这是EI的核心：模型从自己的「成功经验」中学习，相当于自举式生成专家数据。
        Phase 3 SFT：用筛选出的正确数据做一轮监督微调，强化正确的推理模式。

    Args:
        model: 当前策略模型，输入时为train模式，生成时自动切eval，训练时切回
        tokenizer: HF分词器
        device: 训练设备
        train_problems: 训练题目列表，每条包含problem与answer
        reward_fn: 奖励函数，输入(生成回答, 标准答案)，输出得分字典
        n_generations: 每个题目生成的推理轨迹数量
        batch_size: 生成阶段的批次大小（HF时生效）
        sft_epochs: 筛选后数据的SFT训练轮数
        max_new_tokens: 单条生成的最大token数
        temperature: 采样温度，>0启用随机采样，提升生成多样性
        vllm_generator: vLLM生成器（VLLMGenerationConfig实例）。
            提供时用vLLM生成（快），否则用HF generate（慢）

    Returns:
        sft_data: 筛选出的正确样本列表，每条为{prompt, response}
        accuracy: 本轮所有生成轨迹的正确率
    """
    # ========== Phase 1: 生成推理轨迹 (Rollout) ==========
    # 生成任务需要切换到eval模式，关闭dropout等训练层
    model.eval()

    # 构造所有生成任务的prompt与标准答案：每个题目重复n_generations次
    all_prompts: list[str] = []
    all_golds: list[str] = []

    for ex in train_problems:
        # 将原始题目格式化为模型要求的对话prompt
        prompt = make_math_prompt(ex["problem"])
        # 提取标准答案
        gt = ex["answer_raw"] if "answer_raw" in ex else str(ex.get("answer", ""))
        # 每个题目生成n条，对应n次采样
        for _ in range(n_generations):
            all_prompts.append(prompt)
            all_golds.append(gt)

    print(f"  Generating {len(all_prompts)} rollouts "
          f"({len(train_problems)} problems × {n_generations})...")

    # 生成任务必须左填充：保证所有样本的生成起始位置对齐
    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    all_responses: list[str] = []

    if vllm_generator is not None:
        # ── vLLM 高速生成 ──
        all_responses = vllm_generator.generate(
            all_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        # ── HF generate 生成（fallback） ──
        orig_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"

        with torch.inference_mode():
            for i in tqdm(range(0, len(all_prompts), batch_size),
                           desc="  Generating", unit=" batch"):
                batch_ps = all_prompts[i : i + batch_size]
                encoded = tokenizer(
                    batch_ps, return_tensors="pt", padding=True,
                    add_special_tokens=False,
                ).to(device)

                input_len = encoded["input_ids"].shape[1]
                gen_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                new_ids = gen_ids[:, input_len:]
                outputs = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                all_responses.extend(outputs)

        tokenizer.padding_side = orig_padding

    # 校验生成数量与输入数量一致
    assert len(all_responses) == len(all_prompts), (
        f"Generation count mismatch: {len(all_responses)} vs {len(all_prompts)}"
    )

    # ========== Phase 2: 筛选正确轨迹 (Filter) ==========
    # 仅保留回答正确的轨迹，作为下一轮SFT的训练数据
    sft_data: list[dict[str, str]] = []
    n_correct = 0  # 正确轨迹数
    n_total = 0    # 总轨迹数

    for prompt, response, gt in zip(all_prompts, all_responses, all_golds):
        # 调用奖励函数打分
        scores = reward_fn(response, gt)
        # 阈值判断：奖励>0.5视为正确
        is_correct = scores.get("reward", 0.0) > 0.5
        n_total += 1
        if is_correct:
            n_correct += 1
            # 正确样本加入训练集
            sft_data.append({"prompt": prompt, "response": response})

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print(f"  Correct: {n_correct}/{n_total} ({accuracy:.1%})")

    # 边界处理：本轮没有任何正确轨迹，跳过SFT，避免空数据报错
    if not sft_data:
        print("  WARNING: No correct trajectories found! Skipping SFT.")
        return [], accuracy

    # ========== Phase 3: 监督微调 (SFT) ==========
    # 切回训练模式
    model.train()

    # 将筛选出的正确数据写入临时JSONL文件，复用PackedSFTDataset的文件读取接口
    temp_path = Path("/tmp/ei_sft_temp.jsonl")
    with open(temp_path, "w") as f:
        for rec in sft_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 构建打包SFT数据集，最大化显存利用率
    packed_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=temp_path,
        seq_length=1024,
        shuffle=True,
    )
    print(f"  Packed dataset: {len(packed_dataset)} sequences")

    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # 等效批次大小 = 单微批次大小 × 梯度累积步数(4)
    total_batch = batch_size * 4
    # 数据迭代器，一次取出一个等效大批次
    dataloader = iterate_batches(
        packed_dataset, batch_size=total_batch, shuffle=True
    )

    # 计算总训练步数
    steps_per_epoch = max(1, len(packed_dataset) // total_batch)
    total_steps = steps_per_epoch * sft_epochs

    # 逐epoch训练
    for epoch in range(sft_epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"  SFT epoch {epoch+1}/{sft_epochs}", unit=" step")
        for batch in dataloader:
            if n_batches >= steps_per_epoch:
                break
            # 拆分为微批次，用于梯度累积
            microbatches = split_microbatches(batch, grad_accum_steps=4)
            accum_loss = 0.0
            entropy_accum = 0.0
            token_count = 0

            for micro_batch in microbatches:
                input_ids = micro_batch["input_ids"].to(device)
                labels = micro_batch["labels"].to(device)
                # 前向 + 提取对数概率 + 熵（get_response_log_probs 一步完成）
                result = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=True,
                )
                log_probs = result["log_probs"]
                token_entropy = result["token_entropy"]
                # Packed数据集全序列有效，掩码全1
                response_mask = torch.ones_like(log_probs)

                # 计算SFT损失，自动做梯度累积缩放
                loss, meta = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=len(microbatches),
                )
                # 反向传播，梯度累加
                loss.backward()
                accum_loss += meta["loss"].item()

                # 累加熵（用于 wandb 日志）
                if wandb_step is not None:
                    entropy_accum += (token_entropy * response_mask).sum().item()
                    token_count += response_mask.sum().item()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 优化器更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

            epoch_loss += accum_loss
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{accum_loss:.3f}"})

            # wandb 日志：每步 optimizer step 记录一次
            if wandb_step is not None:
                avg_entropy = entropy_accum / max(token_count, 1)
                wandb.log({
                    "train/sft_loss": accum_loss,
                    "train/avg_entropy": avg_entropy,
                }, step=wandb_step[0])
                wandb_step[0] += 1
        pbar.close()

        # 打印本轮平均损失
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{sft_epochs}: avg loss={avg_loss:.4f}")

    # 清理临时文件
    temp_path.unlink(missing_ok=True)

    return sft_data, accuracy


def expert_iteration_loop(args: Any) -> None:
    """
    多轮专家迭代主循环，管理完整的EI训练生命周期。

    执行流程：
        1. 初始化环境、模型、数据集、日志
        2. 逐轮执行「生成→筛选→SFT」迭代
        3. 每轮结束后保存检查点、验证集评估、记录指标
        4. 自动保存最优检查点
        5. 训练结束后输出汇总报告

    Args:
        args: 命令行参数字典，包含模型路径、迭代轮数、生成参数、输出目录等
    """
    # 固定随机种子
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 权重存储到共享存储目录，通过软链接映射到输出目录（工程适配：跨容器共享大文件）
    run_name = f"ei-{Path(args.model_id).name}-g{args.ei_generations_per_prompt}"
    weights_dir = Path("/root/gpufree-share/data/sft/checkpoints") / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 创建软链接，方便从输出目录访问权重
    link_path = output_dir / "checkpoints"
    link_path.unlink(missing_ok=True)
    link_path.symlink_to(Path("../../../gpufree-share/data/sft/checkpoints") / run_name)

    # ── 初始化W&B实验追踪 ──
    wandb.init(
        project=getattr(args, "wandb_project", "cs336-ei-reasoning"),
        entity=getattr(args, "wandb_entity", None),
        name=run_name,
        config={
            "model_id": args.model_id,
            "ei_rounds": args.ei_rounds,
            "ei_generations_per_prompt": args.ei_generations_per_prompt,
            "ei_sft_epochs": args.ei_sft_epochs,
            "ei_temperature": args.ei_temperature,
            "ei_batch_size": args.ei_batch_size,
            "train_data": args.data_path,
        },
    )

    # ── 加载模型与分词器 ──
    dtype = getattr(args, "dtype", "bfloat16")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.to(device)

    # ── 加载训练题目集（MATH训练集） ──
    print(f"Loading training data: {args.data_path}")
    train_examples = load_math_examples(args.data_path, limit=args.ei_train_limit)
    print(f"  {len(train_examples)} training problems")

    # ── 加载验证集，用于每轮评估 ──
    val_examples = load_math_examples(args.val_path, limit=args.eval_limit or 200)
    val_prompts = [make_math_prompt(ex["problem"]) for ex in val_examples]
    val_golds = [ex["answer_raw"] for ex in val_examples]
    print(f"  {len(val_examples)} validation examples")

    # ── vLLM 引擎初始化（可选） ──
    use_vllm = getattr(args, "engine", "hf") == "vllm"
    vllm_device = getattr(args, "vllm_device", None)
    vllm_generator = None
    if use_vllm and vllm_device is not None:
        # vLLM 需要独占 GPU，提取设备号并设置 CUDA_VISIBLE_DEVICES
        vllm_gpu_id = vllm_device.split(":")[-1] if ":" in vllm_device else vllm_device
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu_id)
        print(f"Initializing vLLM on GPU {vllm_gpu_id}...")

        # 保存模型权重到临时目录，供 vLLM 加载
        import tempfile, shutil
        vllm_model_dir = Path("/root/gpufree-share/models/ei-vllm-temp")
        if vllm_model_dir.exists():
            shutil.rmtree(vllm_model_dir)
        model.save_pretrained(vllm_model_dir)
        for fn in ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "tokenizer.model"]:
            src = Path(args.model_id) / fn
            if src.exists():
                shutil.copy2(src, vllm_model_dir / fn)

        from cs336_alignment.eval.generation import VLLMGenerationConfig
        vllm_generator = VLLMGenerationConfig(
            model_id_or_dir=str(vllm_model_dir),
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
        )
        print(f"  vLLM ready on GPU {vllm_gpu_id}")

    # ── 多轮EI主循环 ──
    round_results = []
    start_time = time.perf_counter()
    wandb_step = [0]  # 可变的 step 计数器，传入 round 函数内部递增

    for round_idx in range(1, args.ei_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Expert Iteration Round {round_idx}/{args.ei_rounds}")
        print(f"{'='*60}")

        round_start = time.perf_counter()

        # 执行单轮EI：生成 → 筛选 → SFT
        sft_data, rollout_accuracy = expert_iteration_round(
            model=model,
            tokenizer=tokenizer,
            device=device,
            train_problems=train_examples,
            reward_fn=r1_zero_reward_fn,
            n_generations=args.ei_generations_per_prompt,
            batch_size=args.ei_batch_size,
            sft_epochs=args.ei_sft_epochs,
            max_new_tokens=args.ei_max_new_tokens,
            temperature=args.ei_temperature,
            vllm_generator=vllm_generator,
            wandb_step=wandb_step,
        )

        # SFT 后模型权重变了，需要更新 vLLM
        if vllm_generator is not None:
            print("  Syncing weights to vLLM...")
            import shutil
            model.save_pretrained(vllm_model_dir)
            # 重建 vLLM（加载新权重）
            del vllm_generator
            import gc; gc.collect()
            from cs336_alignment.eval.generation import VLLMGenerationConfig
            vllm_generator = VLLMGenerationConfig(
                model_id_or_dir=str(vllm_model_dir),
                tensor_parallel_size=1,
                dtype="bfloat16",
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
            )

        # 保存本轮筛选出的正确rollout记录
        rollout_path = output_dir / f"rollouts_round{round_idx}.jsonl"
        with open(rollout_path, "w") as f:
            for rec in sft_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Rollouts saved: {rollout_path} ({len(sft_data)} correct)")

        # 保存本轮模型检查点
        ckpt_path = weights_dir / f"round_{round_idx}.pt"
        torch.save({
            "round": round_idx,
            "model_state_dict": model.state_dict(),
            "rollout_accuracy": rollout_accuracy,
            "num_correct": len(sft_data),
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # 在验证集上评估当前模型效果
        eval_metrics = _ei_eval(model, tokenizer, val_prompts, val_golds, device)
        print(f"  Eval: accuracy={eval_metrics['accuracy']:.4f} "
              f"({eval_metrics['correct']}/{eval_metrics['total']})")

        # 写入W&B日志（使用 wandb_step 延续，确保 wandb 能看到连续曲线）
        wandb.log({
            "eval/rollout_accuracy": rollout_accuracy,
            "eval/num_correct": len(sft_data),
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/format_rate": eval_metrics["format_rate"],
            "eval/elapsed_seconds": round(time.perf_counter() - round_start, 1),
            f"round_{round_idx}/rollout_accuracy": rollout_accuracy,
            f"round_{round_idx}/num_correct": len(sft_data),
            f"round_{round_idx}/eval_accuracy": eval_metrics["accuracy"],
            f"round_{round_idx}/eval_format_rate": eval_metrics["format_rate"],
        }, step=wandb_step[0])
        wandb_step[0] += 1

        # 记录本轮结果
        round_elapsed = time.perf_counter() - round_start
        round_results.append({
            "round": round_idx,
            "rollout_accuracy": rollout_accuracy,
            "num_correct": len(sft_data),
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_format_rate": eval_metrics["format_rate"],
            "elapsed_seconds": round(round_elapsed, 1),
        })

        # 自动保存最优检查点（按验证集准确率判断）
        if round_idx == 1 or eval_metrics["accuracy"] > max(
            r["eval_accuracy"] for r in round_results[:-1]
        ):
            best_path = weights_dir / "best.pt"
            torch.save({
                "round": round_idx,
                "model_state_dict": model.state_dict(),
                "eval_accuracy": eval_metrics["accuracy"],
            }, best_path)
            print(f"  ★ New best: {best_path}")

    # ── 训练结束，输出汇总 ──
    total_elapsed = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"  Expert Iteration Complete")
    print(f"{'='*60}")
    print(f"  Rounds:      {args.ei_rounds}")
    print(f"  Generations: {args.ei_generations_per_prompt} per prompt")
    print(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  Round-by-round results:")
    print(f"  {'Round':>5} | {'Rollout Acc':>11} | {'Correct':>7} | {'Eval Acc':>9} | {'Time':>7}")
    print(f"  {'-'*5} | {'-'*11} | {'-'*7} | {'-'*9} | {'-'*7}")
    for r in round_results:
        print(f"  {r['round']:>5} | {r['rollout_accuracy']:>10.1%} | "
              f"{r['num_correct']:>7} | {r['eval_accuracy']:>8.1%} | "
              f"{r['elapsed_seconds']:>6.0f}s")

    # 保存完整结果到文件
    write_json(output_dir / "ei_results.json", {
        "config": {
            "model_id": args.model_id,
            "ei_rounds": args.ei_rounds,
            "ei_generations_per_prompt": args.ei_generations_per_prompt,
            "ei_sft_epochs": args.ei_sft_epochs,
            "ei_temperature": args.ei_temperature,
            "train_data": args.data_path,
        },
        "rounds": round_results,
    })

    wandb.finish()


def _ei_eval(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    val_prompts: list[str],
    val_golds: list[str],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """
    EI专用轻量验证评估函数，逻辑与SFT评估一致，复用数学判分工具。
    
    功能：贪心解码生成答案，提取最终结果并与标准答案比对，计算准确率与格式合规率。
    每轮EI结束后调用，用于跟踪模型推理能力的迭代提升效果。
    """
    model.eval()
    # 生成用左填充
    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    correct = 0
    total = 0
    formatted = 0

    with torch.inference_mode():
        for i in range(0, len(val_prompts), batch_size):
            batch_ps = val_prompts[i : i + batch_size]
            batch_gs = val_golds[i : i + batch_size]
            encoded = tokenizer(
                batch_ps, return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).to(device)
            input_len = encoded["input_ids"].shape[1]
            # 贪心解码，保证评估可复现
            gen_ids = model.generate(
                **encoded, max_new_tokens=512, do_sample=False,
                temperature=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)

            # 逐题判分
            for o, g in zip(outputs, batch_gs):
                pred = extract_math_answer(o)
                if pred is not None:
                    formatted += 1
                    is_correct = grade(pred, g, fast=True)
                else:
                    is_correct = False
                correct += int(is_correct)
                total += 1

    # 恢复原始设置
    tokenizer.padding_side = orig_padding
    model.train()
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "format_rate": formatted / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }