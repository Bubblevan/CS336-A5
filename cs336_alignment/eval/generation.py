"""Generation backends for evaluation.

V0 目标：
1. 保留 HuggingFace generate 作为 fallback；
2. 新增 vLLM offline batched inference；
3. 给 eval/gsm8k.py 一个统一的 generator.generate(prompts, ...) 接口。

为什么需要这个文件？
- baseline eval 不应该和具体推理引擎绑死；
- 之后 GRPO rollout 也可以复用这里的 vLLMGenerator；
- 现在先只做 greedy / temperature=0 的确定性生成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from tqdm import tqdm

def truncate_at_stop_strings(
    text: str,
    stop_strings: list[str] | None,
) -> str:
    """Post-process generated text by cutting at the earliest stop string.

    vLLM 自带 stop 参数；HF generate 没有同样简单的字符串 stop。
    为了两个 backend 行为尽量一致，HF 路径生成后手动截断。
    """
    if stop_strings is None:
        return text

    earliest: int | None = None     # 记录最早出现的 stop string 的位置

    for stop in stop_strings:
        if not stop:
            continue
        idx = text.find(stop)
        # 如果找到了 stop string，并且位置比当前记录的 earliest 更早，就更新 earliest
        if idx >=0 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text
    return text[:earliest]

class TextGenerator(Protocol):
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[str]:
        """Generate text continuations for a batch of prompts."""
        ...
    # 上面的省略号是 Python 3.12 的新语法，表示这个方法没有具体实现，是一个接口定义。

@dataclass
class GenerationConfig:
    """HuggingFace batched generation backend.

    这个比上一版逐条 generate 快很多，但通常仍然慢于 vLLM。
    主要用途：
    - vLLM 暂时不可用时 fallback；
    - 调试模型加载 / Flash Attention；
    - 之后需要 logits/logprobs 时扩展。
    """

    model: torch.nn.Module
    tokenizer: object
    device: torch.device | str
    batch_size: int = 8

    def __post_init__(self) -> None:
        # decoder-only模型通常left padding
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        
        if getattr(self.tokenizer, "pad_token", None) is None:
            # 一些 tokenizer（如 LLaMA）没有 pad_token，设置为 eos_token 以避免报错
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[str]:
        self.model.eval()

        outputs: list[str] = []
        
        for start in tqdm(
            range(0, len(prompts), self.batch_size),
            desc="HF generate",
        ):
            batch_prompts = prompts[start : start + self.batch_size]
            
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(self.device) for key, value in encoded.items()
            }
            # shape[1] 是输入 prompt 的长度，generate 后输出的 shape[1] 是 prompt 长度 + 生成的长度
            input_length = encoded["input_ids"].shape[1]

            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id

            generated_ids = self.model.generate(
                **encoded,      # **encoded 是 Python 的语法糖，表示把 encoded 字典里的 key-value 对作为关键字参数传递给 generate 方法。
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            new_token_ids = generated_ids[:, input_length:]
            batch_outputs = self.tokenizer.batch_decode(
                new_token_ids,
                skip_special_tokens=True,
            )
            batch_texts = [
                truncate_at_stop_strings(text, stop_strings)
                for text in batch_outputs
            ]
            outputs.extend(batch_texts)
        return outputs

@dataclass
class VLLMGenerationConfig:
    """vLLM offline batched generation backend.
    这是 baseline eval 的默认推荐路径。
    vLLM 会在内部做 continuous batching / paged attention，
    比逐条 HuggingFace generate 更适合 GSM8K 这种离线批量评测。
    """
    model_id_or_dir: str
    tensor_parallel_size: int = 1       
    # vLLM 的 tensor parallelism 大小，默认为 1 表示不开启 TP。
    # 作用是 在多卡环境下分散模型参数以节省单卡显存，通常 TP=2 就能显著提升能评测的模型大小了。
    dtype: str = "auto"              # float16, bfloat16, auto
    gpu_memory_utilization: float = 0.9
    max_model_length: int | None = None
    trust_remote_code: bool = True
    seed: int = 0
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    chunk_size: int = 512
    # chunk_size 是 vLLM 生成时的上下文 chunk 大小，即每次处理多少 token 的上下文。较小的 chunk_size 可以减少显存占用，但可能增加生成时间。512 是一个常见的折中选择。
    # 此事在 Continuous Batching 里亦有记载

    def __post_init__(self) -> None:
        # 延迟 import，避免没有 vLLM 时影响 HF backend。
        import vllm

        llm_kwargs = {
            "model": self.model_id_or_dir,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.seed,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            # 避免模型仓库 generation_config 覆盖我们显式传入的采样参数。
            "generation_config": "vllm",
        }
        if self.max_model_length is not None:
            llm_kwargs["max_model_len"] = self.max_model_length

        self.llm = vllm.LLM(**llm_kwargs)

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=stop_strings,
        )
        outputs: list[str] = []

        # 1319 条 GSM8K 一次性丢进去通常也行。
        # 这里保留 chunk，是为了以后更大的 benchmark 不爆内存。
        for start in tqdm(
            range(0, len(prompts), self.chunk_size),
            desc="vLLM generate",
        ):
            chunk_prompts = prompts[start : start + self.chunk_size]
            request_outputs = self.llm.generate(
                chunk_prompts,
                sampling_params=sampling_params,
            )
            
            for request_output in request_outputs:
                text = request_output.outputs[0].text
                # 因为temperature=0且n=1，
                text = truncate_at_stop_strings(text, stop_strings)
                outputs.append(text)
                
        return outputs