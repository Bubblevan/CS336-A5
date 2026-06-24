# core/scoring.py
# -----------------------------------------------------------------------------
# 监督微调（SFT）得分与分布指标计算工具
# 在SFT训练流水线中的定位：
#   分词 → 掩码构建 → 模型前向 → 【本模块：从logits中提取标签对数概率、计算分布熵】 → 损失聚合 → 反向更新
# 核心能力：
#   1. 从模型输出logits中提取真实标签的对数概率（交叉熵损失的核心计算单元）
#   2. 计算每个token位置的预测分布熵，用于训练状态监控与置信度分析
#   3. 封装端到端的前向+得分计算接口，兼容HuggingFace因果语言模型
# -----------------------------------------------------------------------------

from __future__ import annotations  # 类型提示向前兼容

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_entropy(logits: Tensor) -> Tensor:
    """
    计算预测分布的逐token信息熵，公式：H = -Σ p_i * log(p_i)
    
    数值稳定实现推导：
        设词表大小为V，单个位置的logits为 [z_0, z_1, ..., z_{V-1}]
        概率 p_i = exp(z_i) / Z，其中配分函数 Z = Σ exp(z_i)
        因此 log(p_i) = z_i - log(Z)，而 log(Z) = logsumexp(z)
        
        代入熵公式：
        H = -Σ p_i * log(p_i)
          = -Σ p_i * (z_i - log(Z))
          = -Σ p_i z_i + log(Z) * Σ p_i
          = log(Z) - Σ p_i z_i    （因 Σ p_i = 1）
    
    SFT训练中的监控意义：
        - 熵持续过低：模型分布坍缩，只会输出少数几个高频token，可能出现模式崩溃
        - 熵持续过高：模型对预测没有信心，尚未学到有效的生成模式
        - 正常训练：熵随训练步数平稳下降，最终稳定在合理区间

    Args:
        logits: (B, L, V) 模型原始输出logits，B=batch大小，L=序列长度，V=词表大小

    Returns:
        entropy: (B, L) 每个token位置的信息熵，标量非负
    """
    # 步骤1：计算log(Z) = logsumexp(logits)，在词表维度做对数求和指数，数值稳定
    # 输出形状 (B, L)，每个位置对应一个配分函数的对数值
    lse = torch.logsumexp(logits, dim=-1)

    # 步骤2：计算每个token的概率分布 p = softmax(logits)
    # 输出形状 (B, L, V)
    probs = F.softmax(logits, dim=-1)

    # 步骤3：计算期望项 E[logits] = Σ p_i * z_i
    # 元素相乘后在词表维度求和，输出形状 (B, L)
    expectation = torch.sum(probs * logits, dim=-1)

    # 步骤4：代入推导公式 H = log(Z) - E[logits]，得到逐位置熵
    entropy = lse - expectation
    return entropy


def compute_log_probs_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    从模型输出logits中，提取每个位置真实标签对应的对数概率 log p(label | context)
    
    这是SFT交叉熵损失的核心计算单元：
        单token交叉熵损失 = - log p(真实标签)
        批量平均损失 = - masked_mean(log_probs, response_mask)

    实现要点：
        使用 log_softmax 而非 softmax + log，全程在对数域计算，避免概率下溢，数值更稳定
        使用 torch.gather 进行索引提取，高效且支持批量并行计算

    Args:
        logits: (B, L, V) 模型前向输出的原始logits
        labels: (B, L) 真实标签token ID，与logits的序列维度一一对齐

    Returns:
        log_probs: (B, L) 每个位置真实标签的对数概率，取值 ≤ 0
    """
    # 步骤1：对logits在词表维度做log_softmax，得到全词表的对数概率
    # 输出形状 (B, L, V)，每个位置的所有token对应一个log prob
    log_probs_full = F.log_softmax(logits, dim=-1)

    # 步骤2：调整labels维度，适配torch.gather的输入要求
    # (B, L) → (B, L, 1)，增加最后一维，与log_probs_full的维度数一致
    labels_expanded = labels.unsqueeze(-1)

    # 步骤3：按最后一维（词表维）索引，取出每个位置对应真实标签的对数概率
    # gather(dim=-1, index=labels_expanded)：对每个样本、每个位置，取index指定的词表位置的值
    # 输出形状 (B, L, 1)
    gathered = torch.gather(log_probs_full, dim=-1, index=labels_expanded)

    # 步骤4：去掉多余的最后一维，得到 (B, L) 的逐token对数概率
    return gathered.squeeze(-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """
    端到端接口：输入因果LM模型与对齐好的输入/标签，前向传播并返回真实标签的逐token对数概率
    可选同时返回逐token熵，用于训练监控。

    ⚠️ 输入对齐约定：
        传入的 input_ids 与 labels 必须是已经完成移位（shift）的配对序列
        （即 input_ids 是原序列[:-1]，labels 是原序列[1:]）
        与 tokenize_prompt_and_output 的输出格式完全兼容，无需额外处理

    典型使用场景：
        1. SFT训练：配合response_mask计算加权平均损失
        2. 困惑度(PPL)计算：PPL = exp(-平均log_prob)
        3. 样本难度评估：通过log_prob判断哪些样本模型学得不好

    Args:
        model: HuggingFace格式的因果语言模型
        input_ids: (B, L) 移位后的输入token序列
        labels: (B, L) 移位后的真实标签序列
        return_token_entropy: 是否同时返回逐token熵，用于训练监控

    Returns:
        字典，包含：
            "log_probs": (B, L) 真实标签的逐token对数概率
            "token_entropy": (B, L) 可选，逐token预测分布熵
    """
    # 步骤1：模型前向传播，得到输出对象
    output = model(input_ids)

    # 步骤2：取出logits张量，形状 (B, L, V)
    logits = output.logits

    # 步骤3：计算真实标签的对数概率，形状 (B, L)
    log_probs = compute_log_probs_from_logits(logits, labels)

    # 组装返回结果
    result: dict[str, Tensor] = {
        "log_probs": log_probs,
    }

    # 可选：计算并返回逐token熵
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result