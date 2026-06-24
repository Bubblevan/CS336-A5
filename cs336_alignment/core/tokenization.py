# core/tokenization.py
# -----------------------------------------------------------------------------
# 监督微调（Supervised Fine-Tuning, SFT）数据预处理核心工具
# 功能：将「提示文本(prompt) + 输出文本(response)」对，处理为因果语言模型训练可用的张量格式
# 核心设计遵循SFT标准范式：
#   1. 仅对 response 部分计算损失，prompt 部分不参与梯度更新
#   2. 适配因果语言模型的「自回归预测下一词」目标，对序列做一位移位
#   3. 批量序列右填充，支持不同长度的样本组成 batch 训练
# -----------------------------------------------------------------------------

from __future__ import annotations  # Python 类型提示向前兼容，支持类型注解中引用自身

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
# PreTrainedTokenizerBase 是 HuggingFace 所有分词器的基类
# 以此作为类型约束，可兼容 GPT、Llama、Qwen 等所有 HF 格式的分词器


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt and output strings, produce shifted input/labels + response mask.

    Decision chain:
        1. Naive: prompt + response 拼起来 → tokenize → 全部算 loss
        2. 问题：prompt 不该贡献梯度 → 需要 response_mask
        3. 问题：因果 LM 需要 shift（input_ids[:-1], labels[1:]）
        4. 问题：序列长度不同 → 需要 padding

    Returns:
        input_ids:  (B, max_len-1) — shift 后的输入
        labels:     (B, max_len-1) — shift 后的标签
        response_mask: (B, max_len-1) — 1=response token, 0=prompt/padding
    """
    batch_size = len(prompt_strs)
    # 校验输入：prompt 数量与 response 数量必须一一对应
    assert len(output_strs) == batch_size

    all_token_ids: list[list[int]] = []   # 存储每条样本拼接后的完整token id序列
    all_masks: list[list[int]] = []      # 存储每条样本的损失掩码（0=不计损失，1=计损失）
    prompt_lengths: list[int] = []       # 记录每条样本的prompt token长度，预留扩展用

    # 逐样本分词并构建掩码
    for p_str, o_str in zip(prompt_strs, output_strs):
        # 分别对 prompt 和 response 分词，关闭自动添加特殊token
        # 👉 为什么分开分词？
        #    若直接拼接字符串再分词，部分BPE/WordPiece分词器可能产生跨边界的合并token
        #    分开分词可以100%精确控制prompt和response的token边界，保证掩码计算准确
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        
        # 按「prompt在前、response在后」的顺序拼接完整序列
        combined = p_ids + o_ids
        all_token_ids.append(combined)
        prompt_lengths.append(len(p_ids))

        # 构建单条样本的损失掩码
        # 👉 SFT核心逻辑：模型只需要学习生成回答，不需要学习复述用户的提示
        #    0 标记 prompt 部分 → 训练时该位置损失会被掩码掉
        #    1 标记 response 部分 → 训练时该位置正常计算损失
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_masks.append(mask)

    # --------------------------
    # 批量右填充（Right-padding）
    # --------------------------
    # 计算当前batch中最长序列的长度，作为统一填充目标
    max_len_full = max(len(ids) for ids in all_token_ids)
    
    # 获取填充token的id：优先用分词器的pad_token
    # 若分词器没有pad_token（如Llama系列默认无pad），则用eos_token_id替代
    # 这是NLP微调中的通用兼容方案
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 初始化填充张量：全部用pad_id填充，形状 [batch_size, 最大序列长度]
    padded_ids = torch.full((batch_size, max_len_full), pad_id, dtype=torch.long)
    # 初始化掩码张量：全部为0 → 填充部分默认不计入损失
    padded_masks = torch.zeros((batch_size, max_len_full), dtype=torch.long)

    # 逐样本将原始序列填入张量左侧，右侧保留填充值
    # 👉 为什么用右填充？
    #    因果语言模型是从左到右自回归计算，有效内容靠左、填充靠右
    #    不会破坏前缀的注意力计算逻辑，是训练阶段的标准做法
    for i in range(batch_size):
        seq_len = len(all_token_ids[i])
        padded_ids[i, :seq_len] = torch.tensor(all_token_ids[i], dtype=torch.long)
        padded_masks[i, :seq_len] = torch.tensor(all_masks[i], dtype=torch.long)

    # --------------------------
    # 因果LM序列移位（Shift）
    # --------------------------
    # 👉 因果语言模型的训练目标：给定前 t 个token，预测第 t+1 个token
    #    因此输入和标签必须错位一位：
    #       输入 input_ids = [tok0, tok1, ..., tok_{N-2}]  （前N-1个token）
    #       标签 labels    = [tok1, tok2, ..., tok_{N-1}]  （后N-1个token）
    #    这样 input_ids 的第 i 个位置，对应的真实标签就是 labels 的第 i 个位置
    #
    # 损失掩码 response_mask 需要与 labels 严格对齐，因此同样取后N-1位
    # 最终掩码为1的位置，才会参与交叉熵损失计算
    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:].clone()  # clone避免张量共享内存，防止后续意外修改原数据
    response_mask = padded_masks[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def pad_to_max_len(
    token_ids: list[list[int]] | list[torch.Tensor],
    pad_token_id: int,
    max_len: int,
) -> torch.Tensor:
    """
    通用工具：对批量token序列执行**右填充**到指定最大长度
    若原始序列长度超过 max_len，会从左侧开始截断到 max_len
    
    Args:
        token_ids: 批量token序列，支持两种输入格式：嵌套整数列表、torch张量列表
        pad_token_id: 用于填充位置的token ID
        max_len: 统一的目标序列长度
    
    Returns:
        torch.Tensor: 形状为 [batch_size, max_len] 的填充后张量，数据类型为long
    """
    batch_size = len(token_ids)
    # 初始化张量，所有位置先填充为pad_token_id
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    
    for i, seq in enumerate(token_ids):
        # 兼容两种输入格式：列表转张量，张量则直接使用
        seq_t = torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
        # 取序列实际长度与max_len的较小值，实现「超长截断、不足填充」
        length = min(len(seq_t), max_len)
        # 将有效token填入张量左侧
        padded[i, :length] = seq_t[:length]
    return padded


def apply_chat_template(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> str:
    """
    调用分词器内置的对话模板，将对话消息列表格式化为模型可识别的标准字符串
    
    👉 SFT背景说明：
    不同模型有各自的对话格式规范（如Llama 3、ChatML、Qwen模板）
    使用分词器自带的chat_template可以保证格式与预训练阶段完全对齐，避免分布偏移，提升微调效果
    
    Args:
        messages: 对话消息列表，每条消息为字典，包含 role（角色：system/user/assistant）和 content（文本内容）
        tokenizer: 带chat_template配置的HuggingFace分词器实例
    
    Returns:
        str: 格式化后的完整对话字符串，可直接传入分词函数处理
    """
    # tokenize=False 表示仅返回格式化字符串，不执行分词操作
    return tokenizer.apply_chat_template(messages, tokenize=False)