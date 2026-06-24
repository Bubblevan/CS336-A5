# core/masking.py
# -----------------------------------------------------------------------------
# 监督微调（SFT）训练核心掩码工具集
# 核心职能：
#   1. 带掩码的损失聚合：仅对response部分计算平均/归一化损失，屏蔽prompt与padding
#   2. 响应掩码构建：根据prompt长度生成逐位置损失掩码，处理移位后的边界对齐
#   3. 因果注意力掩码：构建下三角掩码，保证自回归模型无法看到未来token
# -----------------------------------------------------------------------------

from __future__ import annotations  # 类型提示向前兼容

import torch
from torch import Tensor


def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """
    对张量中**掩码有效位置（mask=1）**计算平均值，是SFT损失计算的核心工具。
    
    设计背景（SFT训练决策链）：
        1. 朴素做法：对序列所有token的交叉熵损失直接求平均
        2. 问题：prompt部分是用户输入，不需要模型学习复述，padding无意义
        3. 解决方案：用0/1掩码过滤无效位置，仅对response的有效token求平均
        4. 鲁棒性：添加除零保护，避免全无效位置时出现NaN
    
    Args:
        tensor: 待聚合的张量，通常是逐token的损失值，形状 (B, L) 或更高维
        mask:   0/1 掩码张量，形状与tensor对应维度一致，1表示参与计算
        dim:    求平均的维度；None表示对所有元素全局求平均
    
    Returns:
        平均后的张量，标量或指定维度缩减后的张量
    """
    # 第一步：元素级相乘，将mask=0的位置数值清零，等价于过滤掉无效位置
    masked = tensor * mask

    if dim is None:
        # 全局模式：对所有元素求和，统计全部有效token数量
        total = torch.sum(masked)
        count = torch.sum(mask)
    else:
        # 按维度模式：比如按序列维度dim=1求平均，得到每个样本的平均损失
        total = torch.sum(masked, dim=dim)
        count = torch.sum(mask, dim=dim)
    
    # 除零保护：将有效数量的最小值限制为1.0
    # 极端场景（如单样本全为prompt/无有效token）下避免除以0产生NaN
    count = count.clamp(min=1.0)
    
    # 有效位置总和 / 有效位置数量 = 有效位置的平均值
    return total / count


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float | Tensor,
    dim: int | None = None,
) -> Tensor:
    """
    对张量中掩码有效位置求和后，除以**自定义归一化常数**。
    
    与 masked_mean 的核心区别：
        - masked_mean：除数 = 有效token数量（mask.sum()），用于「按token平均损失」
        - masked_normalize：除数 = 外部传入的常数，用于自定义归一化场景
    
    典型使用场景：
        梯度累积训练中，总损失需要除以 gradient_accumulation_steps（梯度累积步数），
        而不是除以当前batch的有效token数，此时就需要用自定义常数做归一化。
    
    Args:
        tensor:              待聚合的张量，形状 (B, L) 或更高维
        mask:                0/1 掩码张量，1表示参与求和
        normalize_constant:  归一化除数（标量或可广播张量）
        dim:                 求和的维度；None表示全局求和
    
    Returns:
        归一化后的张量
    """
    # 先过滤无效位置
    masked = tensor * mask

    if dim is None:
        # 全局求和后除以归一化常数
        return torch.sum(masked) / normalize_constant
    # 按指定维度求和后除以归一化常数
    return torch.sum(masked, dim=dim) / normalize_constant


def build_response_mask(
    input_ids: Tensor,
    prompt_lengths: list[int] | Tensor,
) -> Tensor:
    """
    根据prompt长度构建响应掩码：response位置为1，prompt与padding位置为0。
    
    ⚠️ 核心易错点（CS336高频考点）：
        传入的 input_ids 已经完成了因果LM的**移位操作**（去掉了最后一个token），
        而 prompt_lengths 是**移位前**的原始prompt token长度。
        因此移位后，response的起始索引需要 -1 才能正确对齐。

    边界推导示例：
        原始完整序列（移位前，长度 N = p + r）：
            索引:    0   1  ...  p-1   p    p+1  ...  p+r-1
            内容:  [p0, p1, ..., p_end, r0,  r1,  ..., r_end]
                     ←—— prompt部分 ——→ ←—— response部分 ——→
        
        移位后的 labels 序列（长度 N-1）：取原始序列的 [1:]
            索引:    0   1  ...  p-2   p-1    p    ...  p+r-2
            内容:  [p1, p2, ..., p_end,  r0,    r1,   ..., r_end]
        
        结论：原始response的第一个token(r0)，在移位后的序列中索引为 p-1
             因此掩码中response的起始位置 = prompt_length - 1

    Args:
        input_ids:      (B, L) 移位后的输入/标签张量，用于获取batch大小、序列长度和设备
        prompt_lengths: 每个样本移位前的prompt token长度
    
    Returns:
        mask: (B, L) 0/1掩码，1表示该位置属于response，参与损失计算
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device  # 与输入同设备，避免CPU/GPU张量不匹配报错

    # 初始化全0掩码，默认所有位置都不参与损失计算
    mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    for i in range(batch_size):
        p_len = int(prompt_lengths[i])
        # 关键：移位后response的起始索引 = 原始prompt长度 - 1
        response_start = p_len - 1
        
        # 边界保护：防止prompt过长、response为空时索引越界
        if response_start < seq_len:
            # 从response起始位置到序列末尾，全部置为1（有效损失位置）
            mask[i, response_start:] = 1

    return mask


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """
    构建因果（自回归）注意力掩码：下三角全1矩阵。
    
    作用原理：
        因果语言模型的核心约束是「每个token只能看到自己和之前的token，不能看到未来的token」。
        该掩码形状为 (seq_len, seq_len)，对角线及以下位置为1，以上为0。
        在自注意力计算中，会将掩码为0的位置对应的注意力分数置为 -inf，
        经过softmax后权重趋近于0，从而实现对未来信息的屏蔽。

    Args:
        seq_len: 序列长度
        device:  张量所在设备（CPU/GPU）
    
    Returns:
        (seq_len, seq_len) 的下三角浮点张量
    """
    # torch.tril = lower triangular，保留矩阵下三角部分（含对角线），上三角置为0
    return torch.tril(torch.ones(seq_len, seq_len, device=device))