"""Parsing helpers for benchmark outputs.

V0 目标：
- GSM8K：从模型输出中抽取最后一个数字；
- GSM8K gold：优先从 "#### 42" 后面抽答案；
- MMLU：先给一个极简 A/B/C/D parser，后续再完善。
"""

from __future__ import annotations
# __future__ 意思是从未来版本导入特性，这里是为了启用 Python 3.10+ 的类型提示功能。

import re
from decimal import Decimal, InvalidOperation

_NUMBER_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# 这个正则表达式用于匹配数字，包括整数、浮点数和科学计数法表示的数字。

def normalize_number_str(text: str) -> str | None:
    # 这个函数用于规范化数字字符串，去掉逗号和下划线，以便后续转换为数字。
    matches = _NUMBER_REGEX.findall(text)
    if not matches:
        return None

    number = matches[-1]  # 取最后一个匹配的数字
    number = number.replace(",", "")
    number = number.strip() # 去掉前后空白

    if number.endswith("."):
        number = number[:-1]  # 去掉末尾的点
    
    if number in {"", "+", "-"}:
        return None
    return number

def parse_last_number(text: str | None) -> str | None:
    if not text:
        return None
    return normalize_number_str(text)

def parse_gsm8k_gold_answer(answer: str) -> str | None:
    # 这个函数用于从 GSM8K gold 的模型输出中抽取答案，优先从 "#### 42" 后面抽答案。
    if not answer:
        return None
    
    answer = str(answer)
    
    if "####" in answer:
        # 如果模型输出中包含 "####"，我们就优先从 "#### 42" 后面抽取数字。
        tail = answer.split("####", maxsplit=1)[-1]
        parsed = parse_last_number(tail)
        # 如果 "####" 后面能成功抽取到数字，就直接返回这个数字；
        # 否则就继续从整个文本中抽取最后一个数字。
        if parsed is not None:
            return parsed
    
    # 如果没有找到 "#### 42"，就直接从整个文本中抽取最后一个数字
    return parse_last_number(answer)

def parse_gsm8k_response(model_output: str) -> str | None:
    
    if model_output is None:
        return None
    text = str(model_output)

    # 我们更偏向显式答案标签
    # 因为SFT/GRPO的prompt会要求
    answer_tag_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text, 
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_tag_matches:
        # 如果模型输出中包含 "<answer>...</answer>" 标签，我们就优先从标签内抽取数字。
        parsed = parse_last_number(answer_tag_matches[-1])
        if parsed is not None:
            return parsed

    if "####" in text:
        # 这种GSM8K风格也要支持
        tail = text.split("####", maxsplit=1)[-1]
        parsed = parse_last_number(tail)
        # 如果 "####" 后面能成功抽取到数字，就直接返回这个数字；
        # 否则就继续从整个文本中抽取最后一个数字。
        if parsed is not None:
            return parsed

    # 如果没有找到 "<answer>...</answer>" 标签或者 "#### 42"，就直接从整个文本中抽取最后一个数字
    return parse_last_number(text)

# 经典等价性判断
def numbers_equal(
    pred: str | None,
    gold: str | None,
    *,  # 这个星号表示后面的参数必须以关键字参数的形式传入，不能作为位置参数。
    atol: float = 1e-6,
) -> bool:
    if pred is None or gold is None:
        return False

    try:
        pred_num = Decimal(pred)
        gold_num = Decimal(gold)
    except InvalidOperation:
        # 如果无法转换为数字，就直接比较字符串是否相等（忽略大小写和前后空白）
        return pred.strip().lower() == gold.strip().lower()

    # 如果成功转换为数字，就比较它们是否在给定的绝对误差范围内相等。
    return abs(pred_num - gold_num) <= Decimal(atol)

def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    # 这个函数用于从 MMLU 的模型输出中抽取答案，先给一个极简 A/B/C/D parser，后续再完善。
    if not model_output:
        return None
    
    text = str(model_output).strip()
    # 我们更偏向显式答案标签
    answer_tag_matches = re.findall(
        r"(?:answer|final answer|choice)\s*[:=]?\s*([A-D])\b",
        text, 
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_tag_matches:
        # 如果模型输出中包含 "answer: A" 或 "final answer = B" 之类的显式答案标签
        # 我们就优先从标签内抽取字母。
        return answer_tag_matches[-1].upper()

    # 否则就直接从整个文本中抽取最后一个字母。
    matches = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None