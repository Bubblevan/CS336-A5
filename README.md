# CS336 Spring 2025 — Assignment 5: Alignment

本作业实现语言模型的后训练（Post-Training）对齐流程，包括：

1. **SFT（Supervised Fine-Tuning）** — 在推理轨迹上进行监督微调
2. **Expert Iteration** — 基于策略生成 + 奖励筛选的迭代式训练
3. **GRPO（Group Relative Policy Optimization）** — 基于组归一化奖励的策略梯度优化
4. **DPO（Direct Preference Optimization）** — 直接偏好优化（可选补充作业）

## 项目结构

```
cs336-a5/
├── cs336_alignment/           # 主包
│   ├── __init__.py
│   ├── SFT_util/              # SFT 工具函数
│   │   ├── __init__.py
│   │   ├── compute_entropy_func.py   # 熵、log-probs、masked normalize
│   │   ├── tokenize_func.py          # prompt+output tokenization
│   │   ├── train_step.py             # SFT micro-batch train step + log_generations
│   │   └── model_test.py             # 模型评估
│   ├── prompts/               # Prompt 模板
│   │   ├── r1_zero.prompt
│   │   ├── alpaca_sft.prompt
│   │   ├── question_only.prompt
│   │   └── zero_shot_system_prompt.prompt
│   ├── tests/                 # 单元测试（来自 handout 的 snapshot 测试）
│   │   ├── __init__.py
│   │   ├── adapters.py        # 测试适配器（连接实现与测试）
│   │   ├── test_sft.py        # SFT 相关测试
│   │   ├── test_grpo.py       # GRPO 相关测试
│   │   ├── test_data.py       # 数据加载测试
│   │   ├── test_metrics.py    # 评价指标测试
│   │   └── test_dpo.py        # DPO 测试
│   ├── run_sft.py             # SFT 训练主脚本
│   ├── run_grpo.py            # GRPO/Expert Iteration 训练主脚本
│   ├── drgrpo_grader.py       # MATH 打分函数（格式 + 答案判分）
│   └── plot_sft_curves.py     # 训练曲线可视化
├── data/                      # 数据集
│   ├── alpaca_eval_hf/        # AlpacaEval 评估基准
│   ├── gsm8k/                 # GSM8K 数学推理基准
│   ├── hh/                    # Anthropic HH-RLHF 偏好数据
│   ├── mmlu_hf/               # MMLU 多任务语言理解基准
│   └── simple_safety_tests/   # 简单安全测试集
├── models/                    # 预训练模型权重（gitignored）
├── outputs/                   # 训练输出（checkpoints、日志）
├── logs/                      # 运行日志
├── slurm/                     # SLURM 作业脚本
│   ├── run_sft.slurm
│   └── run_grpo.slurm
├── pyproject.toml             # 项目配置 + 依赖
├── CHANGELOG.md
└── README.md
```

## 环境搭建

本作业使用 **uv** 管理依赖和 Python 环境（Python 3.12）。

### 初始化（首次 / 依赖变更后）

```bash
# 安装 uv（如果还没有）
apt update && apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 安装项目依赖
# uv sync --no-install-package flash-attn
uv sync
```
这个 Flash Attention 确实很难搞。确定你当前环境的PyTorch版本、CUDA版本、Python版本
```bash
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvcc --version  # 检查系统CUDA编译器版本
python --version
python -c "import torch; print('ABI TRUE' if torch.compiled_with_cxx11_abi() else 'ABI FALSE')"
```
之后去flash-attn的[GitHub Release页面](https://github.com/Dao-AILab/flash-attention/releases)，下载一个文件名完全匹配你环境的.whl文件吧。对我现在这个环境来说是`uv pip install /path/to/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl。`
### 日常使用

大部分操作可以用 `uv run` 自动使用项目环境，无需手动 activate：

```bash
# 运行 Python 脚本（自动使用 uv 环境）
uv run python -m cs336_alignment.run_benchmarks --help

# 进入交互式 Python
uv run python

# 在 uv 环境里执行任意命令
uv run bash

# 运行测试
uv run pytest
```

**手动激活虚拟环境**（例如安装本地 wheel、调试依赖时）：

```bash
source .venv/bin/activate
```

## 数据准备

本作业使用 5 个数据集。下载方式见各小节，本地数据已就绪，存放在 `data/` 下。

### 模型权重

```bash
./hfd.sh Qwen/Qwen2.5-Math-1.5B \
  --local-dir /root/gpufree-share/models/Qwen2.5-Math-1.5B \
  -x 8 -j 6
```

> 如果从 HuggingFace 拉取失败，确认已配置 `huggingface-cli login` 或使用镜像站。

### GSM8K

- **来源**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) — OpenAI 发布的 8,500 道小学数学应用题，每道需 2-8 步推理。
- **许可**: MIT
- **格式**: Parquet（`test-*.parquet` / `train-*.parquet`）
- **数量**: train 7,473 / test 1,319（每个 variant 一样）
- **数据字段**: `question`（数学题文本） / `answer`（逐步推理 + 最终答案）

两个子集：

| 子集 | 目录 | Answer 格式特点 |
|------|------|----------------|
| `main` | `data/gsm8k/main/` | 标准 CoT，推理 + `<<expr=result>>` 计算标注 + `#### 42` 最终答案 |
| `socratic` | `data/gsm8k/socratic/` | 苏格拉底自问自答式，如 `How many X? ** ... \n#### 42` |

示例（`main`）:
```
Q: Natalia sold clips to 48 of her friends in April, and then she sold
   half as many clips in May. How many clips did Natalia sell altogether?
A: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
   Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
#### 72
```

```bash
./hfd.sh openai/gsm8k \
  --dataset \
  --local-dir data/gsm8k
```

### MMLU

- **来源**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) — Massive Multitask Language Understanding，57 个学科的多选题基准。
- **许可**: 见 [repository](https://github.com/hendrycks/test)（非商业用途）
- **格式**: HuggingFace datasets（每科目一个子目录）
- **数量**: 约 14,042 test / 1,531 val / 285 dev / 99,842 auxiliary_train
- **数据字段**: `question`（题目） / `choices`（4 个选项列表） / `answer`（正确答案字母 A-D）

覆盖 57 个科目，包括 elementary_mathematics、us_history、computer_science、law 等，评价模型跨领域知识。

```bash
./hfd.sh cais/mmlu \
  --dataset \
  --local-dir data/mmlu_hf
```

### HH-RLHF（Anthropic Helpful & Harmless）

- **来源**: [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) — 人类偏好数据，用于训练 Reward Model / RLHF。
- **许可**: MIT
- **格式**: JSONL gzip（每行 `{"chosen": "...", "rejected": "..."}`）
- **内容**: 对话数据，包含 5 个子集：

| 子集 | 目录 | 说明 |
|------|------|------|
| `helpful-base` | `data/hh/helpful-base/` | 基于 base model 的有用性偏好 |
| `helpful-online` | `data/hh/helpful-online/` | 迭代在线 RLHF 采样数据 |
| `helpful-rejection-sampled` | `data/hh/helpful-rejection-sampled/` | Rejection sampling 数据 |
| `harmless-base` | `data/hh/harmless-base/` | 无害性偏好数据 |
| `red-team-attempts` | `data/hh/red-team-attempts/` | 红队攻击对话记录（含评分、标签） |

```bash
./hfd.sh Anthropic/hh-rlhf \
  --dataset \
  --local-dir data/hh
```

### AlpacaEval

- **来源**: [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) — 自动化 LLM 评估基准，基于 AlpacaFarm 的 805 条指令。
- **许可**: CC-BY-NC-4.0
- **格式**: JSON（`alpaca_eval.json`，805 条）
- **数据字段**: `instruction`（指令） / `output`（参考输出） / `generator`（生成器标识） / `dataset`（来源）
- **用途**: 评估模型对开放性指令的回复质量（通常用 GPT-4 / 自动 judge 打分）

```bash
./hfd.sh tatsu-lab/alpaca_eval \
  --dataset \
  --local-dir data/alpaca_eval_hf
```

### SimpleSafetyTests

- **来源**: [Bertievidgen/SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) — 100 条关键安全风险测试用例。
- **许可**: CC-BY-2.0
- **格式**: CSV（`sst_test_cases.csv`，100 条 prompt）
- **危害类别**: 自杀/自残、人身伤害、非法/管制物品、诈骗、儿童虐待
- **用途**: 快速评估模型是否拒绝有害请求。正常模型应对全部 100 条 prompt 都拒绝回答。

```bash
./hfd.sh Bertievidgen/SimpleSafetyTests \
  --dataset \
  --local-dir data/simple_safety_tests
```

## 训练流程

### SFT
```bash
uv run python -m cs336_alignment.run_sft \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --device cuda:0 \
    --vllm_device cuda:1 \
    --max_steps 200 \
    --batch_size 8
```

### GRPO / Expert Iteration
```bash
uv run python -m cs336_alignment.run_grpo \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --device cuda:0 \
    --vllm_device cuda:1 \
    --group_size 8 \
    --max_steps 100
```

## 基准评估

### GSM8K（数学推理）

使用 `run_benchmarks.py` 对模型进行 GSM8K 测试集评估：

```bash
uv run python -m cs336_alignment.run_benchmarks \
    --model_id /root/gpufree-share/models/Qwen2.5-Math-1.5B \
    --benchmarks gsm8k \
    --gsm8k_path data/gsm8k/main \
    --output_dir outputs/baseline_qwen_math \
    --device cuda:0 \
    --max_new_tokens 512
```

参数说明：
- `--model_id` — 模型路径或 HuggingFace ID
- `--benchmarks` — 评估基准（当前仅支持 `gsm8k`）
- `--gsm8k_path` — 数据文件或目录。接受 `data/gsm8k`、`data/gsm8k/main` 或具体的 parquet/jsonl 文件
- `--output_dir` — 输出目录（生成 `summary.json` 和 `gsm8k_predictions.jsonl`）
- `--limit N` — 仅跑前 N 条做快速验证
- `--seed` — 随机种子（默认 0）

输出示例：
```
GSM8K summary:
  benchmark: gsm8k
  split: test
  num_examples: 1319
  correct: 372
  accuracy: 0.2820
  parsed: 1310
  parsed_ratio: 0.9932
```

## 提交

```bash
bash make_submission.sh
```
