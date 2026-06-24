# core/batching.py
# -----------------------------------------------------------------------------
# 监督微调（SFT）数据加载与批处理模块
# 在SFT训练流水线中的定位：
#   原始数据 → 【本模块：数据集构建、样本打包、批处理拆分】 → 模型前向 → 得分计算 → 损失反向
# 核心能力：
#   1. Packed SFT 数据集：将多段对话拼接后切分为固定长度序列，最大化GPU利用率，减少padding浪费
#   2. 批次迭代器：按指定batch size生成训练批次，支持打乱顺序
#   3. 微批次拆分：实现梯度累积所需的micro-batch切分，用小显存模拟大batch训练效果
#   4. 样本拼接函数：将单样本张量堆叠为批次张量
# -----------------------------------------------------------------------------

from __future__ import annotations  # 类型提示向前兼容

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase


class PackedSFTDataset(Dataset):
    """
    打包式SFT数据集（Packed SFT Dataset），继承自PyTorch标准Dataset接口。
    
    设计背景与核心思想：
        普通SFT数据按单条对话为样本，长度参差不齐，组成batch时需要大量padding，
        导致GPU计算资源被无意义的填充token浪费，显存利用率低。
        Packed 方案：将所有对话的token首尾拼接成一条超长扁平序列，
        再按固定长度 seq_length 切成一个个块（chunk），每个块都是满长度的有效序列，
        几乎没有padding，显著提升训练吞吐。

    移位约定：
        数据集在预处理阶段就完成因果LM的移位对齐：
        每个块取 seq_length + 1 个连续token，前 seq_length 个作为 input_ids，
        后 seq_length 个（整体右移一位）作为 labels，训练时无需再次移位。

    Args:
        tokenizer: HuggingFace分词器实例
        dataset_path: 数据集路径，格式为jsonl，每行包含 prompt 和 response 字段
        seq_length: 每个训练样本的固定序列长度
        shuffle: 是否在打包前打乱文档顺序
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool = False,
    ):
        self.seq_length = seq_length

        # --------------------------
        # 步骤1：加载原始数据集
        # --------------------------
        # 按行读取jsonl文件，跳过空行，每行解析为一个字典
        with open(dataset_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # 打包前打乱文档顺序，避免同主题数据集中在一起，提升训练稳定性
        if shuffle:
            random.shuffle(lines)

        # --------------------------
        # 步骤2：全量分词并拼接成扁平token序列
        # --------------------------
        # 存储所有对话拼接后的token id 长序列
        all_tokens: list[int] = []
        eos_id = tokenizer.eos_token_id

        for item in lines:
            # 取出单条样本的提示与回答
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            # 直接拼接 prompt + response（数据已由蒸馏脚本用 r1_zero 模板格式化）
            text = prompt + response

            # 分词，add_special_tokens=True 会自动在开头添加BOS(句子开始)标记
            # ⚠️ 注意：当前实现将prompt与response整体分词并计入训练序列
            # 若需严格仅response算损失，需额外记录每段的prompt长度并构建对应mask
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(ids)

            # 每条样本末尾追加EOS标记，分隔不同对话，让模型学习句子边界
            if eos_id is not None:
                all_tokens.append(eos_id)

        # --------------------------
        # 步骤3：按固定长度切分样本，同时完成移位对齐
        # --------------------------
        # 预处理好所有样本，存储为张量列表，__getitem__直接返回即可
        self.examples: list[dict[str, Tensor]] = []

        # 确定填充token id：优先pad_token，回退到eos_token，最终回退到0
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id or 0)

        # 按 seq_length 步长滑动切分
        # 为什么取 seq_length + 1 个 token？
        # → 同时容纳 input_ids（前 seq_length 位）和 labels（后 seq_length 位，右移一位）
        idx = 0
        while idx < len(all_tokens):
            # 取 seq_length 个 token 作为输入
            input_chunk = all_tokens[idx : idx + seq_length]
            actual_len = len(input_chunk)

            # 不足时 padding
            if actual_len < seq_length:
                pad_len = seq_length - actual_len
                input_chunk = input_chunk + [pad_id] * pad_len
                # labels: 右移一位，padding 位置预测 pad_id
                label_chunk = all_tokens[idx + 1 : idx + actual_len] + [pad_id] * (pad_len + 1)
            else:
                # 正常情况：取下一个 token 作为最后一个位置的 label
                if idx + seq_length < len(all_tokens):
                    label_last = all_tokens[idx + seq_length]
                else:
                    label_last = pad_id
                label_chunk = input_chunk[1:] + [label_last]

            self.examples.append({
                "input_ids": torch.tensor(input_chunk[:seq_length], dtype=torch.long),
                "labels": torch.tensor(label_chunk[:seq_length], dtype=torch.long),
            })

            idx += seq_length

    def __len__(self) -> int:
        """返回数据集总样本数，PyTorch Dataset 标准接口"""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """根据索引获取单条样本，PyTorch Dataset 标准接口"""
        return self.examples[idx]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool = False,
) -> Dataset:
    """
    工厂函数：快速构建Packed SFT数据集实例，对外提供简洁调用接口。
    
    封装底层类的实例化逻辑，便于后续扩展（如添加缓存、多格式支持）而不修改调用方代码。

    Args:
        tokenizer: 分词器实例
        dataset_path: 数据集文件路径
        seq_length: 固定序列长度
        shuffle: 是否打乱文档顺序

    Returns:
        构建好的Dataset实例
    """
    return PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle,
    )


def collate_fn(batch_examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    样本拼接函数（Collate Function）：将多条单样本拼接为一个批次张量。

    适用场景：
        PackedSFTDataset产出的每个样本已经是固定长度的张量，
        因此无需动态padding，直接在第0维（batch维）堆叠即可。
        这也是packed数据集效率高的原因之一——batch组装开销极低。

    Args:
        batch_examples: 单样本字典组成的列表，每个字典包含input_ids、labels等键

    Returns:
        批次字典，每个键对应形状为 (B, seq_length) 的张量
    """
    result: dict[str, Tensor] = {}
    # 遍历所有字段，分别做batch维度的堆叠
    for key in batch_examples[0].keys():
        result[key] = torch.stack([ex[key] for ex in batch_examples], dim=0)
    return result


class BatchIterator:
    """支持 len() 和迭代的批次迭代器。"""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_examples = [self.dataset[i] for i in batch_indices]
            yield collate_fn(batch_examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
):
    """
    批次迭代器：按指定批次大小遍历数据集，逐个返回训练批次。

    设计说明：
        返回 BatchIterator 实例，支持 len() 查询总批次数，
        也支持 for 循环逐批迭代。相比 PyTorch DataLoader 更轻量透明。

    Args:
        dataset: PyTorch Dataset实例
        batch_size: 每个批次包含的样本数量
        shuffle: 是否在每个epoch开始时打乱样本顺序

    Returns:
        BatchIterator 实例，支持 len() 和迭代
    """
    return BatchIterator(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def split_microbatches(
    batch: dict[str, Tensor],
    grad_accum_steps: int,
) -> list[dict[str, Tensor]]:
    """
    将一个完整批次拆分为多个微批次（Micro-batch），用于梯度累积训练。

    梯度累积原理：
        当显存不足以容纳大batch时，将大batch拆成N个小的micro-batch，
        逐个执行前向+反向传播，梯度累加N次后再执行一次优化器更新，
        最终等效于大batch的训练效果，是大模型训练的常用显存优化手段。

    拆分规则：
        在batch维度（第0维）均匀切分，最后一个micro-batch允许略小；
        批次内所有字段同步切分，保证样本对应关系一致。

    Args:
        batch: 完整批次字典，张量形状均为 (B, ...)
        grad_accum_steps: 梯度累积步数，即拆分的微批次数量

    Returns:
        微批次字典列表，每个微批次张量形状为 (micro_batch_size, ...)
    """
    # 取第一个字段的第0维大小，得到完整批次的样本数
    batch_size = batch[list(batch.keys())[0]].shape[0]

    # 计算每个微批次的样本数，向上取整保证所有样本都被分配
    micro_size = math.ceil(batch_size / grad_accum_steps)

    microbatches: list[dict[str, Tensor]] = []
    # 按micro_size步长切分所有字段
    for start in range(0, batch_size, micro_size):
        end = min(start + micro_size, batch_size)
        # 对每个字段都做相同范围的切片
        microbatch = {key: tensor[start:end] for key, tensor in batch.items()}
        microbatches.append(microbatch)

    return microbatches