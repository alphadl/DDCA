# DCA: Decoupled Conditional Advantage for Efficient Reasoning

DCA 复现代码与脚本。本仓库**不包含**完整 RL 训练框架，仅提供 DCA 优势计算，供在现有 GRPO/RLOO 中替换；基于 [verl](https://github.com/verl-project/verl) 时见 [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md)。

---

## 一键复现论文结果

**环境**（一次性）：

```bash
cd efficient_reason_DCA
pip install -r requirements.txt
```

**一键运行**（准备数据 → 演示推理 → 评估，约几秒）：

```bash
BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh
```

输出：pass@1、avg_tokens 等；数据与结果在 `data/processed/`。

**可选环境变量**（有默认值）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATA_DIR` | `data/processed` | 数据与结果目录 |
| `TRAIN_SIZE` / `VAL_SIZE` | 200 / 50 | 样本数（HF 可用时） |
| `BUILTIN_ONLY` | 0 | 1=仅内置样本，免下载（快速演示） |
| `USE_MATH` | 0 | 1=加入 MATH（约 1:2） |
| `RESULTS_FILE` | 空 | 指定已有结果 JSONL 则只跑评估 |

**用 VERL 复现**：先 `python scripts/prepare_data.py --output_dir data/processed [--builtin_only]` 得到 `train.parquet`，再按 [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md) patch VERL 后运行 `./scripts/run_verl_baselines.sh`。已有 VERL 结果时：`RESULTS_FILE=/path/to/results.jsonl ./scripts/run_full_pipeline.sh` 仅做评估。

---

## 方法简述

- **问题**：GRPO 等易导致推理过长；奖励中简单加长度惩罚易崩。
- **原因**：混合组中错误样本拉低均值（基线稀释）；全对组方差归一化使长度系数失效。
- **方法**：DCA——正确性与效率解耦，仅在正确集内算长度优势，用 Sigmoid 映射为有界分数。

---

## 项目结构

```
efficient_reason_DCA/
├── dca/                 # 优势计算、指标、数据工具、VERL 接口
├── scripts/
│   ├── run_full_pipeline.sh   # 一键：准备→演示→评估
│   ├── prepare_data.py       # 小规模数据（parquet + jsonl）
│   ├── demo_inference.py     # 演示推理（无 VERL 时）
│   ├── evaluate.py           # pass@1、avg_tokens、AES
│   ├── run_verl_baselines.sh # VERL 三 baseline 对照
│   ├── run_verl_comparison.py # 本地对比三种 advantage
│   ├── verify_dca.py         # 公式校验
│   ├── cpu_mini_validate.py  # 极小规模 DCA vs 耦合 LP
│   └── train_dca.py          # DCA API 检查（dry_run）
├── configs/experiment.yaml   # 实验配置
├── configs/verl/             # VERL 配置片段
└── tests/
```

---

## 进阶

- **公式与性质**：`python scripts/verify_dca.py`
- **极小验证（CPU）**：`python scripts/cpu_mini_validate.py`
- **评估**：`python scripts/evaluate.py --results path/to/results.jsonl --k 1`
- **单元测试**：`python scripts/run_tests.py`

**论文实验设置**：Qwen3-1.7B / DeepSeek-R1-Distill-Qwen-1.5B；AIME+MATH ~1:2、2500 条；GSM8K/MATH500/AMC/AIME 评估；pass@1、avg_tokens、AES；β≈0.2。

---

## License

研究使用；模型与数据遵循各自许可。
