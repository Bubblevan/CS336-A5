"""
reasoning/prompts.py

Prompt 模板加载与格式化工具。

功能：
    - load_r1_zero_prompt_template(path)：加载 r1_zero.prompt（<think>/<answer> 格式）
    - load_question_only_prompt_template(path)：加载 question_only.prompt（\\boxed{} 格式）
    - format_math_prompt(problem, template)：将题目填入模板

决策链：
    1. 朴素做法：每个训练/评估脚本各自写 prompt 拼接代码
    2. 问题：prompt 格式散落在各处，改模板需要改多处代码
    3. 所以：把模板加载和格式化统一到 promps.py，所有模块从这里导入

模板格式约定：
    r1_zero.prompt 使用 {question} 占位符：
        "User: {question}\nAssistant: <think>"

    question_only.prompt 也使用 {question} 占位符：
        "{question} Please put your final answer within \\boxed{{}}."
"""

from __future__ import annotations

from pathlib import Path


def load_r1_zero_prompt_template(path: str | Path) -> str:
    """
    加载 r1_zero.prompt 模板。

    模板特点：
        - DeepSeek-R1-Zero 风格的 <think>/<answer> 格式
        - 包含 System + User + Assistant 的多轮对话结构
        - 用 {question} 作为题目占位符

    Args:
        path: promp 文件路径（绝对或相对路径）

    Returns:
        str: 模板内容，包含 {question} 占位符
    """
    return Path(path).read_text(encoding="utf-8")


def load_question_only_prompt_template(path: str | Path) -> str:
    """
    加载 question_only.prompt 模板。

    模板特点：
        - 仅含题目本身，不附带多轮对话结构
        - 答案格式为 \\boxed{{}}（LaTeX boxed 环境）
        - 用 {question} 作为题目占位符

    适用场景：
        - 基线评估（zero-shot 对比）
        - 消融实验（对比 r1_zero 格式的效果）

    Args:
        path: promp 文件路径

    Returns:
        str: 模板内容，包含 {question} 占位符
    """
    return Path(path).read_text(encoding="utf-8")


def format_math_prompt(problem: str, template: str) -> str:
    """
    将数学题目填入 prompt 模板。

    简单封装 str.format(question=problem)，
    提供统一的调用入口，便于后续扩展（如转义特殊字符）。

    Args:
        problem:  数学题目字符串
        template: 包含 {question} 占位符的模板

    Returns:
        str: 格式化后的完整 prompt
    """
    return template.format(question=problem)
