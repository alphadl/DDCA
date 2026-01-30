# VERL 接入 DCA

本仓库提供统一 advantage 接口（vanilla / grpo_lp / dca），在 [verl](https://github.com/verl-project/verl) 中安装本库并 patch 一处即可切换。

## 安装与 patch

```bash
cd efficient_reason_DCA && pip install -e .
```

在 VERL 中**计算 advantage 的位置**（搜索 `rewards`、`mean`/`std` 或 `grpo`）将：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

改为：

```python
from dca.verl_integration import compute_advantage
adv_mode = getattr(config.algorithm, "adv_mode", None) or "vanilla"
if adv_mode in ("vanilla", "grpo_lp", "dca", "dca_rloo"):
    advantages = compute_advantage(
        rewards, batch["response_lengths"],
        correct_mask=None, mode=adv_mode,
        beta=getattr(config, "beta", 0.2),
        gamma=getattr(config, "gamma", 1e-3),
        use_rloo=(adv_mode == "dca_rloo"),
    )
else:
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

Reward 侧：vanilla/dca 用 0/1；grpo_lp 用 `(1 - gamma*length)` if correct else 0。可用 `reward_for_verl(correct, lengths, mode=adv_mode, gamma=gamma)`。

## 运行对照

```bash
# 本地对比三种 advantage 输出（不装 VERL）
python scripts/run_verl_comparison.py

# 实际跑 VERL 三个 baseline（需已 patch + 数据）
RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_verl_baselines.sh [overrides]
```

配置片段见 `configs/verl/`（vanilla / grpo_lp / dca）。
