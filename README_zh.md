# DCA: Decoupled Conditional Advantage for Efficient Reasoning

本仓库为 DCA（Decoupled Conditional Advantage）方法的复现代码与脚本。

## 方法要点

- **问题**：RLVR（如 GRPO）易导致推理冗长（overthinking）；在奖励中简单加入长度惩罚常导致效果崩溃。
- **原因**：  
  1. **长度基线稀释（Dilution of Length Baseline）**：混合质量组中，错误样本无长度惩罚，拉低组均值，使正确样本被错误地相对“罚长”。  
  2. **参数失效（Parameter Inefficacy）**：全对组中方差归一化会使长度惩罚系数 γ 在公式中被抵消，无法调节长度。
- **方法**：**DCA（Decoupled Conditional Advantage）**  
  - 将正确性与效率解耦；  
  - 仅在**正确样本集合**内计算长度优势（条件化），消除基线稀释；  
  - 用 Sigmoid 将长度 Z-score 映射为有界分数，效率作为“信息密度”的正向奖励而非单纯惩罚。

## 环境与依赖

```bash
cd efficient_reason_DCA
pip install -r requirements.txt
```

## 项目结构

```
efficient_reason_DCA/
├── dca/
│   ├── advantage.py   # DCA-GRPO / DCA-RLOO 优势计算
│   ├── metrics.py    # AES, pass@K
│   ├── data_utils.py # 数据加载与答案等价性
│   └── __init__.py
├── scripts/
│   ├── train_dca.py   # 训练接入说明与 DCA API 检查
│   ├── evaluate.py    # 评估：accuracy, tokens, pass@K, AES
│   └── verify_dca.py  # 公式与性质校验
├── configs/
│   └── experiment.yaml
├── tests/
│   ├── test_advantage.py
│   └── test_metrics.py
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 校验 DCA 公式与性质

```bash
python scripts/verify_dca.py
```

验证：参数失效、长度优势在正确集内零和、DCA 与耦合奖励在混合组上的差异等。

### 2. 在训练循环中使用 DCA

在 GRPO/RLOO 中，将原来的优势替换为 DCA 即可：

```python
from dca import advantage_dca_grpo  # 或 advantage_dca_rloo

# 每个 prompt 采样 G 条回复，得到 correct_mask 与 lengths
correct_mask = np.array([...])  # shape (G,), bool
lengths = np.array([...])       # shape (G,), int

advantages = advantage_dca_grpo(correct_mask, lengths, beta=0.2)
# 用 advantages 做 policy gradient（如 GRPO 的 clip loss）
```

- **GRPO**：`advantage_dca_grpo(correct_mask, lengths, beta)`  
- **RLOO**：`advantage_dca_rloo(correct_mask, lengths, beta)`  

推荐 **β ≈ 0.2** 作为效率-准确率折中。

### 3. 评估

结果文件需为 JSONL，每行一条样本，包含例如：

- `predictions`: 列表（多 rollout）或单个字符串  
- `lengths`: 对应 token 长度列表或单个值  
- `ground_truth` 或 `answer`: 标准答案  

```bash
python scripts/evaluate.py --results path/to/results.jsonl --k 3
# 若提供基线结果，可算 AES
python scripts/evaluate.py --results results.jsonl --base_results base.jsonl --k 3
```

### 4. 单元测试

```bash
python scripts/run_tests.py
# 或（需安装 pytest）：pytest tests/ -v
```

## 实验设置

- **模型**：Qwen3-1.7B、DeepSeek-R1-Distill-Qwen-1.5B  
- **训练数据**：AIME + MATH，约 1:2 混合，共 2500 条  
- **评估**：GSM8K、MATH500、AMC23、AIME25  
- **超参**：temperature=0.6, top_p=0.95, max_tokens=16384；GSM8K/MATH500 每题 3 rollout，AMC/AIME 每题 10 rollout（pass@1/pass@10）  
- **指标**：pass@1、平均 token 数、**AES（Accuracy-Efficiency Score）**

## 仓库推送

首次关联远程并推送到 GitHub（main 分支）：

```bash
git remote add origin https://github.com/alphadl/Decoupled_Conditional_Advantage.git
git branch -M main
git push -u origin main
```

## License

本复现仅供研究使用；模型与数据请遵循各原始来源的许可。
