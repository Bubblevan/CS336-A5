"""GSM8K baseline evaluation.

V0 功能：
- 读取 GSM8K parquet / jsonl；
- 对每道题 greedy generate；
- 从模型输出中解析最终数字；
- 与 gold answer 比较；
- 输出 summary dict；
- 可选保存逐题 predictions.jsonl。

暂时不做：
- batched generation
- few-shot prompting
- vLLM
- distributed eval
- majority vote / self-consistency
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from cs336_alignment.core.utils import append_jsonl, load_jsonl
from cs336_alignment.eval.parsers import parse_gsm8k_response, parse_gsm8k_gold_answer, numbers_equal

def make_gsm8k_prompt(question: str) -> str:
    """Build a simple zero-shot GSM8K prompt.

    这个 prompt 故意简单：
    baseline 阶段先看 base model 的零样本能力。
    后面可以再加 few-shot / chat template。
    """
    return (
        "Solve the following grade school math problem. "
        "Reason step by step, and put only the final numeric answer "
        "inside <answer>...</answer>.\n\n"
        f"Problem:\n{question}\n\n"
        "Solution:\n"
    )

def _read_parquet(path: Path) -> list[dict[str, Any]]:
    # 不知道传闻是不是真的
    # datasets里有pyarrow吗
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.to_pylist()

def _find_gsm8k_file(path: Path, split: str) -> Path:
    """
    GSM8K 的数据文件命名和组织方式比较混乱，这个函数尝试按照一定的优先级顺序找到正确的文件：
    1. 如果用户直接传了一个文件路径（而不是目录），就直接用这个文件。
    2. 如果用户传了一个目录，我们先在这个目录下找一个叫 "main" 的子目录，如果有的话，就在这个子目录里找符合 "{split}-*.parquet" 或 "{split}.jsonl" 命名模式的文件。
    3. 如果在 "main" 子目录里没有找到，我们就在用户传的目录下直接找符合 "{
    """
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"GSM8K path does not exist: {path}")

    candidates: list[Path] = []

    # User may pass data/gsm8k, and the useful subset is data/gsm8k/main.
    main_dir = path / "main"
    if main_dir.is_dir():
        candidates.extend(sorted(main_dir.glob(f"{split}-*.parquet")))
        candidates.extend(sorted(main_dir.glob(f"{split}.jsonl")))

    # User may pass data/gsm8k/main.
    candidates.extend(sorted(path.glob(f"{split}-*.parquet")))
    candidates.extend(sorted(path.glob(f"{split}.jsonl")))

    # Last resort: any parquet/jsonl in that directory.
    candidates.extend(sorted(path.glob("*.parquet")))
    candidates.extend(sorted(path.glob("*.jsonl")))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find GSM8K {split} parquet/jsonl under: {path}"
        )

    return candidates[0]

def load_gsm8k_examples(path: Path, *, split: str="test") -> list[dict[str, Any]]:
    gsm8k_file = _find_gsm8k_file(Path(path), split=split)
    print(f"Loading GSM8K {split} examples from: {gsm8k_file}")
    
    if gsm8k_file.suffix == ".parquet":
        return _read_parquet(gsm8k_file)
    elif gsm8k_file.suffix == ".jsonl":
        return load_jsonl(gsm8k_file)
    else:
        raise ValueError(f"Unsupported GSM8K file format: {gsm8k_file}")



@torch.inference_mode()# 这个装饰器可以让整个函数在推理模式下运行，禁用梯度计算和一些其他的训练相关功能，从而节省内存和提高性能。
def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device | str,
    max_new_tokens: int = 512,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,  # GSM8K 的 prompt 已经是纯文本了，不需要 tokenizer 自动添加特殊 tokens 了。
    )

    encoded = {
        key: value.to(device) for key, value in encoded.items()
    }

    input_length = encoded["input_ids"].shape[1]

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = eos_token_id
    
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding
        temperature=None,  # greedy decoding
        top_p=None,  # greedy decoding
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )[0]

    new_token_ids = output_ids[input_length:]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return output_text


def run_gsm8k_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    gsm8k_path: str | Path,
    *,
    split: str = "test",
    device: torch.device | str,
    max_new_tokens: int = 512,
    output_path: str | Path | None = None,
    limit: int | None = None, # 这个参数可以让我们在调试阶段只 eval 前 N 个样本，避免每次都跑完整个测试集。
) -> dict[str, Any]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    examples = load_gsm8k_examples(gsm8k_path, split=split)

    if limit is not None and limit > 0:
        examples = examples[:limit]

    if not examples:
        raise ValueError("No GSM8K examples found.")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 清空
        output_path.write_text("", encoding="utf-8")

    total = 0
    correct = 0
    parsed = 0
    gold_parsed = 0

    progress = tqdm(examples, desc="Evaluating GSM8K", unit="ex")

    for idx, example in enumerate(progress):
        question = example["question"]
        gold = example["answer"]

        prompt = make_gsm8k_prompt(question)
        model_output = generate_one(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        pred = parse_gsm8k_response(model_output)
        gold_answer = parse_gsm8k_gold_answer(gold)

        is_correct = numbers_equal(pred, gold_answer)

        total += 1
        if is_correct:
            correct += 1
        if pred is not None:
            parsed += 1
        if gold_answer is not None:
            gold_parsed += 1

        progress.set_postfix({
            "acc": f"{correct}/{total}={correct/total:.4f}",
            "parsed": f"{parsed}/{total}={parsed/total:.4f}",
            "gold_parsed": f"{gold_parsed}/{total}={gold_parsed/total:.4f}",
        })

        if output_path is not None:
            append_jsonl(output_path, {
                "question": question,
                "gold": gold,
                "model_output": model_output,
                "pred": pred,
                "gold_parsed": gold_parsed,
                "is_correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        parsed_ratio = parsed / total if total > 0 else 0.0
        gold_parsed_ratio = gold_parsed / total if total > 0 else 0

    return {
        "benchmark": "gsm8k",
        "split": split,
        "data_path": str(gsm8k_path),
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "parsed": parsed,
        "parsed_ratio": parsed_ratio,
        "gold_parsed": gold_parsed,
        "max_new_tokens": max_new_tokens,
    }