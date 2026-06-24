# reasoning/sft.py
# -----------------------------------------------------------------------------
# CS336 对齐算法 - 监督微调（SFT）核心训练与评估模块
# 在完整SFT训练流水线中的定位：
#   数据加载 → 分词掩码 → 模型前向 → 【本模块：损失计算 + 梯度缩放 + 反向传播】 → 优化器更新
#   训练过程中定期调用评估函数：生成回答 → 奖励打分 → 指标统计 → 结果持久化
# 核心能力：
#   1. 标准SFT交叉熵损失计算，仅对response区域计算，兼容梯度累积缩放
#   2. 模型生成效果评估：批量生成、奖励打分、指标聚合、结果落盘
# -----------------------------------------------------------------------------

from __future__ import annotations  # 类型提示向前兼容

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

# 导入前序核心模块：掩码归一化工具、对数概率计算工具
from cs336_alignment.core.masking import masked_normalize
from cs336_alignment.core.scoring import compute_log_probs_from_logits


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Any]]:
    """
    SFT单微批次训练步：计算response区域的交叉熵损失，完成梯度累积所需的损失缩放。
    
    设计决策链（CS336 SFT核心考点）：
        1. 朴素做法：对所有token直接计算交叉熵损失并求平均
        2. 问题：prompt与padding不应贡献梯度 → 用response_mask过滤有效位置
        3. 问题：梯度累积场景下，损失需要按累积步数缩放 → 除以grad_accum_steps
        4. 问题：不同批次有效token数差异大，需支持自定义归一化常数 → normalize_constant参数化

    ⚠️ 关键区分：
        - scaled_loss：已缩放的损失，用于调用 .backward() 做反向传播，携带计算图梯度
        - metadata["loss"]：未缩放的原始微批次平均损失，仅用于日志记录，已detach无梯度

    Args:
        policy_log_probs: (B, L) 模型输出的真实标签逐token对数概率，由 compute_log_probs_from_logits 得到
        response_mask: (B, L) 0/1掩码，1表示response有效位置，仅该区域计算损失
        gradient_accumulation_steps: 梯度累积步数，即多少个微批次执行一次优化器更新
        normalize_constant: 自定义归一化除数，用于损失的尺度校准

    Returns:
        scaled_loss: 缩放后的损失张量，可直接调用 backward()
        metadata: 包含未缩放的原始损失等统计信息，用于日志记录
    """
    batch_size = policy_log_probs.shape[0]

    # Number of valid (unmasked) tokens
    num_valid_tokens = response_mask.sum().clamp(min=1)

    # Step 1: per-token NLL
    nll_per_token = -policy_log_probs

    # Step 2: sum over valid tokens, divide by valid count → per-token average NLL
    total_masked_loss = torch.sum(nll_per_token * response_mask)
    microbatch_loss_mean = total_masked_loss / num_valid_tokens

    # Step 3: scale for gradient accumulation
    scaled_loss = microbatch_loss_mean / gradient_accumulation_steps

    # Return unscaled loss for logging, scaled loss for backward
    return scaled_loss, {"loss": microbatch_loss_mean.detach()}


def log_generations(
    prompts: list[str],
    ground_truths: list[str],
    vllm_model: Any,
    policy_model: torch.nn.Module,
    tokenizer: Any,
    sampling_params: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    save_jsonl_path: str | Path | None = None,
    save_metrics_path: str | Path | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """
    SFT/DPO训练中的验证评估函数：批量生成回答、奖励打分、统计指标并持久化。
    
    功能定位：
        训练过程中定期在验证集上执行生成评估，监控模型的生成质量、答题准确率、格式合规性等指标，
        是判断模型是否过拟合、对齐效果是否提升的核心依据。

    组件说明：
        - vllm_model：vLLM加速推理引擎，用于高速批量生成，比原生HF模型生成快数倍至数十倍
        - policy_model：训练中的PyTorch模型，评估前需要将训练权重同步到vLLM中（通常在函数外部执行）
        - reward_fn：奖励打分函数，输入(生成文本, 标准答案)，输出多维度奖励分数
        - sampling_params：vLLM的采样参数，控制温度、top_p、最大生成长度等生成配置

    Args:
        prompts: 验证集提示文本列表
        ground_truths: 验证集标准答案列表，与prompts一一对应
        vllm_model: vLLM的LLM实例，用于批量生成
        policy_model: 训练中的HF格式策略模型
        tokenizer: HuggingFace分词器
        sampling_params: vLLM SamplingParams 实例，配置生成参数
        reward_fn: 奖励函数，返回包含 reward / format_reward / answer_reward 的字典
        save_jsonl_path: 若设置，将每条样本的生成详情保存为JSONL文件
        save_metrics_path: 若设置，将聚合后的评估指标保存为JSON文件

    Returns:
        records: 每条样本的详细记录列表，包含prompt、生成文本、标准答案、各维度得分
        metrics: 聚合后的评估指标字典，包含准确率、平均奖励、平均生成长度等
    """
    # 注意：训练权重 → vLLM的同步操作通常在函数外部执行
    # 例如：vllm_model.sync_model_weights(policy_model.state_dict())

    # --------------------------
    # 步骤1：批量生成回答
    # --------------------------
    # 调用vLLM的generate接口，一次性输入所有prompt，并行生成
    outputs = vllm_model.generate(prompts, sampling_params)

    # --------------------------
    # 步骤2：逐样本打分与记录
    # --------------------------
    records = []
    # 累计各维度奖励总分，用于后续求平均
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    # 记录每条生成的长度，统计平均输出长度
    response_lengths = []

    for i, output in enumerate(outputs):
        # 取出第一条生成结果（vLLM支持n条采样，这里取第一条）
        generated_text = output.outputs[0].text
        # 取出对应标准答案，边界保护防止列表长度不一致
        gold_answer = ground_truths[i] if i < len(ground_truths) else ""

        # 调用奖励函数，得到多维度打分
        # 通常包含：
        #   - reward: 总奖励（综合得分，可直接用于准确率统计）
        #   - format_reward: 格式合规奖励（例如是否按要求输出JSON、步骤是否完整）
        #   - answer_reward: 答案准确性奖励（与标准答案匹配程度）
        scores = reward_fn(generated_text, gold_answer)
        r = scores.get("reward", 0.0)
        fmt_r = scores.get("format_reward", 0.0)
        ans_r = scores.get("answer_reward", 0.0)

        # 累加总分
        total_reward += r
        total_format_reward += fmt_r
        total_answer_reward += ans_r
        response_lengths.append(len(generated_text))

        # 组装单条记录，便于后续错误分析与实验追溯
        records.append({
            "prompt": prompts[i],
            "generated_text": generated_text,
            "ground_truth": gold_answer,
            "reward": r,
            "format_reward": fmt_r,
            "answer_reward": ans_r,
        })

    # --------------------------
    # 步骤3：计算聚合评估指标
    # --------------------------
    n = len(prompts)
    metrics = {
        # 平均总奖励，通常可直接视为准确率（奖励为0/1时完全等价）
        "eval/accuracy": total_reward / n if n > 0 else 0.0,
        # 平均格式奖励，监控模型输出格式的合规性
        "eval/avg_format_reward": total_format_reward / n if n > 0 else 0.0,
        # 平均答案奖励，监控模型答题准确性
        "eval/avg_answer_reward": total_answer_reward / n if n > 0 else 0.0,
        # 平均生成长度，监控模型是否出现输出变短/变长的偏移
        "eval/avg_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
    }

    # --------------------------
    # 步骤4：结果持久化
    # --------------------------
    # 保存逐样本详情为JSONL，便于后续抽样分析bad case
    if save_jsonl_path:
        save_path = Path(save_jsonl_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 保存聚合指标为JSON，便于实验看板读取与历史对比
    if save_metrics_path:
        metrics_path = Path(save_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return records, metrics