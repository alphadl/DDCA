# Integrating DCA with VERL

This repository **does not depend on or import verl**. The `dca.verl_integration` module exposes an interface (advantage / reward) that **VERL calls**. Install this package inside your [verl](https://github.com/verl-project/verl) environment and add a single patch in VERL’s code (e.g. `from dca.verl_integration import compute_advantage`) to switch between vanilla / grpo_lp / dca.

## Installation and patch

```bash
cd DDCA && pip install -e .
```

In VERL, locate the place where **advantages are computed** (search for `rewards`, `mean`/`std`, or `grpo`). Replace:

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

with:

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

**Reward side:** vanilla/dca use 0/1; grpo_lp uses `(1 - gamma*length)` if correct else 0. You can use `reward_for_verl(correct, lengths, mode=adv_mode, gamma=gamma)`.

## Running baselines

```bash
# Local comparison of the three advantage modes (no VERL required)
python scripts/run_verl_comparison.py

# Run VERL with all three baselines (requires patch + data)
RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_verl_baselines.sh [overrides]
```

Config snippets are in `configs/verl/` (vanilla / grpo_lp / dca).
