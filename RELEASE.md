# 发布说明 (Release Notes)

## 复现内容

本仓库复现 DCA（Decoupled Conditional Advantage）方法的算法与实验设置。

### 已实现

1. **DCA 核心**
   - `advantage_dca_grpo`：DCA-GRPO 优势（式 10、14、15）
   - `advantage_dca_rloo`：DCA-RLOO 优势（式 16、17）
   - 条件长度分数：Z-score 仅在正确集内，Sigmoid 映射（式 12、13）
   - 耦合奖励基线：`rewards_coupled_lp` 用于对比与参数失效验证

2. **指标**
   - AES（Accuracy-Efficiency Score，式 18）
   - pass@K（式 19）
   - 平均 token 数

3. **数据与评估**
   - GSM8K / MATH / 通用 JSONL 加载
   - 数学答案等价性（normalize、boxed、####）
   - 评估脚本：`scripts/evaluate.py`（支持 JSONL 结果与基线对比算 AES）

4. **校验与测试**
   - `scripts/verify_dca.py`：**参数失效**、**长度基线稀释**、DCA 零和性质
   - `tests/`：单元测试（advantage、metrics）

5. **配置与接入**
   - `configs/experiment.yaml`：训练/评估配置占位
   - `scripts/train_dca.py`：说明如何将 DCA 接入现有 GRPO/RLOO 训练循环（`--dry_run` 可检查 API）

### 使用方式摘要

- **仅用 DCA 优势**：在现有 GRPO/RLOO 代码中，将原来的 advantage 替换为  
  `advantage_dca_grpo(correct_mask, lengths, beta=0.2)` 或 `advantage_dca_rloo(...)`。
- **完整训练**：需自行接入 DeepSeek-R1 / Qwen3 等模型与 DAPO/自定义 GRPO 训练框架；本仓库提供 DCA 计算与评估管线。
- **评估**：准备 JSONL（含 `predictions`、`lengths`、`ground_truth`），运行  
  `python scripts/evaluate.py --results <path> [--base_results <base>] [--k 3]`。

### 依赖

- Python ≥ 3.7
- `numpy`、`pyyaml`；测试无需 `pytest`（可用 `python scripts/run_tests.py`）。

### 版本

- 初始复现版本，与 DCA 方法中的公式与实验设置对齐。
