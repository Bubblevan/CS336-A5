# CS336 Spring 2025 — Assignment 5: Alignment

本作业实现语言模型的后训练（Post-Training）对齐流程，包括：

1. **SFT（Supervised Fine-Tuning）** — 在推理轨迹上进行监督微调
2. **Expert Iteration** — 基于策略生成 + 奖励筛选的迭代式训练
3. **GRPO（Group Relative Policy Optimization）** — 基于组归一化奖励的策略梯度优化
4. **DPO（Direct Preference Optimization）** — 直接偏好优化（可选补充作业）

## 项目结构

```
cs336-linux/
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
├── data/                      # 数据集（需从远程服务器下载或挂载）
│   ├── datasets/
│   │   └── sft-reason/        # SFT 推理数据
│   ├── MATH/                  # MATH 验证集
│   ├── MMLU/                  # MMLU 基准
│   └── GSM8K/                 # GSM8K 基准
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

## 环境搭建（在远程 Linux 服务器上）

```bash
# 使用 uv 管理依赖
apt update && apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync --no-install-package flash-attn
uv sync

# 运行测试
uv run pytest
```

## 数据准备
```bash
./hfd.sh Qwen/Qwen2.5-Math-1.5B \
  --local-dir models/Qwen2.5-Math-1.5B \
  -x 8 -j 6
```
将作业 handout 提供的数据放到 `data/datasets/sft-reason/`、`data/MATH/` 等目录下。
```bash
./hfd.sh openai/gsm8k \
  --dataset \
  --local-dir data/gsm8k

./hfd.sh cais/mmlu \
  --dataset \
  --local-dir data/mmlu_hf

./hfd.sh Anthropic/hh-rlhf \
  --dataset \
  --local-dir data/hh

./hfd.sh tatsu-lab/alpaca_eval \
  --dataset \
  --local-dir data/alpaca_eval_hf

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

## 提交

```bash
bash make_submission.sh
```
