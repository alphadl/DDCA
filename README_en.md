# DCA: Decoupled Conditional Advantage for Efficient Reasoning

This repository contains the implementation and scripts for the DCA (Decoupled Conditional Advantage) method.

## Key Ideas

- **Problem**: RLVR (e.g., GRPO) tends to cause overthinking—excessively long reasoning; naively adding length penalty to the reward often leads to performance collapse.
- **Causes**:  
  1. **Dilution of Length Baseline**: In mixed-quality groups, incorrect responses (with no length penalty) lower the group mean, so correct responses are wrongly penalized for being “longer” relative to a distorted baseline.  
  2. **Parameter Inefficacy**: When all responses in a group are correct, variance normalization cancels out the length-penalty coefficient γ in the formula, making length uncontrollable.
- **Method**: **DCA (Decoupled Conditional Advantage)**  
  - Decouple correctness and efficiency;  
  - Compute length advantage **only within the set of correct responses** (conditional), removing baseline dilution;  
  - Map length Z-score to a bounded score via Sigmoid, so efficiency acts as a positive bonus for information density rather than a blunt penalty.

## Setup

```bash
cd efficient_reason_DCA
pip install -r requirements.txt
```

## Project Structure

```
efficient_reason_DCA/
├── dca/
│   ├── advantage.py   # DCA-GRPO / DCA-RLOO advantage computation
│   ├── metrics.py     # AES, pass@K
│   ├── data_utils.py  # Data loading and answer equivalence
│   └── __init__.py
├── scripts/
│   ├── train_dca.py   # Training integration and DCA API check
│   ├── evaluate.py    # Evaluation: accuracy, tokens, pass@K, AES
│   └── verify_dca.py  # Formula and property verification
├── configs/
│   └── experiment.yaml
├── tests/
│   ├── test_advantage.py
│   └── test_metrics.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Verify DCA formulas and properties

```bash
python scripts/verify_dca.py
```

Checks: parameter inefficacy, zero-sum length advantage over the correct set, and DCA vs coupled reward in mixed groups.

### 2. Use DCA in your training loop

In GRPO/RLOO, replace the standard advantage with DCA:

```python
from dca import advantage_dca_grpo  # or advantage_dca_rloo

# For each prompt, sample G responses and get correct_mask and lengths
correct_mask = np.array([...])  # shape (G,), bool
lengths = np.array([...])       # shape (G,), int

advantages = advantage_dca_grpo(correct_mask, lengths, beta=0.2)
# Use advantages in your policy gradient (e.g., GRPO clip loss)
```

- **GRPO**: `advantage_dca_grpo(correct_mask, lengths, beta)`  
- **RLOO**: `advantage_dca_rloo(correct_mask, lengths, beta)`  

Recommended **β ≈ 0.2** for the efficiency–accuracy trade-off.

### 3. Evaluation

Results should be in JSONL format, one sample per line, with fields such as:

- `predictions`: list of strings (multiple rollouts) or a single string  
- `lengths`: list of token counts per rollout or a single value  
- `ground_truth` or `answer`: reference answer  

```bash
python scripts/evaluate.py --results path/to/results.jsonl --k 3
# With baseline results, compute AES
python scripts/evaluate.py --results results.jsonl --base_results base.jsonl --k 3
```

### 4. Unit tests

```bash
python scripts/run_tests.py
# Or (requires pytest): pytest tests/ -v
```

## Experimental setup

- **Models**: Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B  
- **Training data**: AIME + MATH, ~1:2 mix, 2500 samples  
- **Evaluation**: GSM8K, MATH500, AMC23, AIME25  
- **Hyperparameters**: temperature=0.6, top_p=0.95, max_tokens=16384; 3 rollouts per problem for GSM8K/MATH500, 10 for AMC/AIME (pass@1/pass@10)  
- **Metrics**: pass@1, average tokens, **AES (Accuracy-Efficiency Score)**

## Pushing to GitHub

First-time setup and push (main branch):

```bash
git remote add origin https://github.com/alphadl/Decoupled_Conditional_Advantage.git
git branch -M main
git push -u origin main
```

## License

This implementation is for research use only; models and data follow their respective licenses.
