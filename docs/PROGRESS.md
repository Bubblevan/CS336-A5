# PROGRESS — CS336 Assignment 5 (SFT Phase)

## 已完成

### 数据准备（2025-06-23）

- [x] **数据转换脚本**: `scripts/prepare_sft_data.py`
  - 从 `/root/gpufree-share/data/` 读取原始 parquet
  - 输出 JSON + JSONL（两种格式）
  - GSM8K: r1_zero 模板格式化 prompt，提取 response 为 `<think>`/`<answer>` 格式
  - TULU-3: 从 `messages` list 拆出 user→assistant 对

- [x] **GSM8K 转换完成**
  - `data/gsm8k/train.json` — 7,473 条
  - `data/gsm8k/test.json` — 1,319 条
  - 字段: `question`, `answer`, `prompt`, `response`

- [x] **TULU-3 SFT Personas Math 转换完成**
  - `data/tulu-3-sft-personas-math/train.json` — 149,960 条
  - 字段: `question`, `answer`(空), `prompt`, `response`

- [x] **README 更新**: 补充 TULU-3 数据说明 + SFT 格式转换说明
- [x] **PROGRESS 文档**: 本文件

### 复用的现有代码

- `eval/parsers.py` — GSM8K 答案抽取 ✅
- `eval/gsm8k.py` — GSM8K 评估管线 ✅
- `eval/generation.py` — HF / vLLM 生成后端 ✅
- `core/utils.py` — 模型加载、JSON 工具 ✅
- `reasoning/rewards.py` — r1_zero_reward_fn ✅
- `data/gsm8k/test.json` — 验证集，仅 1,319 条，用于 eval

## 待完成

### [ ] 核心层实现（core/）

- `core/tokenization.py` — `tokenize_prompt_and_output`
- `core/masking.py` — `masked_mean`, `masked_normalize`
- `core/scoring.py` — `compute_entropy`, `get_response_log_probs`
- `core/batching.py` — `iterate_batches`, `collate_fn`

### [ ] 推理层实现（reasoning/）

- `reasoning/sft.py` — `sft_microbatch_train_step`, `log_generations`
- `reasoning/train_step.py` — `grpo_microbatch_train_step`
- `reasoning/prompts.py` — prompt template loading

### [ ] 主入口实现

- `run_reasoning_sft.py` — SFT 训练主脚本

### [ ] 验证

- `pytest tests/` — 跑 handout snapshot 测试

## 数据规模

| 数据集 | 训练集 | 测试集 | 总 token 数（估计） |
|--------|--------|--------|-------------------|
| GSM8K (SFT) | 7,473 | 1,319 | ~8M (train) |
| TULU-3 Math | 149,960 | — | ~150M (train) |
