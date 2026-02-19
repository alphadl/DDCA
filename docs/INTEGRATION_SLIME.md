# Integrating DCA with Slime

**Slime repository:** [THUDM/slime](https://github.com/THUDM/slime) (LLM post-training framework for RL scaling, Megatron + SGLang).

This repository **does not depend on or import slime**. The `dca.slime_integration` module exposes an interface (advantage / reward) that **Slime calls**. Install this package in your [THUDM/slime](https://github.com/THUDM/slime) environment and apply the patch below to switch between vanilla / grpo_lp / dca.

## Installation

Install this package in the Slime environment (or clone it and add `DDCA` to `PYTHONPATH`):

```bash
cd /path/to/DDCA && pip install -e .
```

## Slime data layout (THUDM/slime)

- **Sample** (`slime.utils.types.Sample`): single sample with `reward` (float or dict) and `response_length` (int).
- **Train batch:** built from `list[list[Sample]]`; the batch typically has `rewards` and a length array (key may be `response_length` or `response_lengths` depending on the conversion). This package’s `compute_advantage_for_slime` supports custom keys via `reward_key` and `length_key`.

## Patch instructions

### 1. Advantage computation

In Slime, find where **GRPO advantages** are computed (where rewards are normalized to produce advantages). Replace:

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

with:

```python
from dca.slime_integration import compute_advantage_for_slime

adv_mode = getattr(args, "adv_mode", None) or "vanilla"
if adv_mode in ("vanilla", "grpo_lp", "dca", "dca_rloo"):
    # batch must contain rewards and length array; keys must match Slime train data
    advantages = compute_advantage_for_slime(
        batch,
        adv_mode=adv_mode,
        beta=getattr(args, "beta", 0.2),
        gamma=getattr(args, "gamma", 1e-3),
        use_rloo=(adv_mode == "dca_rloo"),
        reward_key="rewards",
        length_key="response_lengths",  # or "response_length", match Slime’s train data key
        group_size=args.n_samples_per_prompt,
    )
else:
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

If Slime uses a custom entry point (e.g. `--custom-pg-loss-reducer-function-path`), you can call `compute_advantage_for_slime` there and return the advantages.

### 2. Reward side

In Slime’s reward function (e.g. a custom RM specified by `--custom-rm-path`), produce a scalar reward by mode:

- **vanilla / dca:** 0/1 (correctness only); length is handled in the DCA advantage.
- **grpo_lp:** (1 - gamma*length) if correct else 0.

You can use this package’s interface:

```python
from dca.slime_integration import reward_for_slime

# In custom RM or reward post-processing, for each sample:
correct = ...   # whether the answer is correct
length = sample.response_length
reward = reward_for_slime(np.array([correct]), np.array([length]), mode=adv_mode, gamma=gamma)[0]
```

If Slime’s `--custom-reward-post-process-path` is used for batch reward post-processing, you can call `reward_for_slime` there to produce 0/1 or coupled rewards for the default or DCA advantage.

### 3. Command line / config

Add to Slime’s launch arguments or config, for example:

- `--adv-mode dca` (or vanilla / grpo_lp)
- `--beta 0.2` (for DCA)
- `--gamma 0.001` (for grpo_lp)

Exact argument names follow THUDM/slime’s `slime.utils.arguments`. If `adv_mode` is not yet defined, read it in the patch from an environment variable or custom config.

## Running all three baselines

```bash
# Local comparison of the three advantage modes (no Slime required)
python scripts/run_verl_comparison.py

# From Slime repo root, after patch and data/config are set, run all three (add Slime args as needed):
RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_slime_baselines.sh [your Slime args...]
```

`run_slime_baselines.sh` runs `SLIME_CMD` (default `python train.py`) and sets `ADV_MODE=vanilla`, then `grpo_lp`, then `dca`. In the Slime advantage patch, read `os.environ.get("ADV_MODE", "vanilla")` and pass it to `compute_advantage_for_slime(..., adv_mode=...)`. If Slime is started via a script (e.g. `bash scripts/run-glm4-9B.sh`), set:

```bash
export SLIME_CMD="bash scripts/run-glm4-9B.sh"
./scripts/run_slime_baselines.sh
```

and pass `adv_mode`, `beta`, etc. via environment or extra arguments inside that script.

## Data preparation

Same as the main README: `python scripts/prepare_data.py --output_dir data/processed [--builtin_only]` produces `train.parquet` etc. Slime may expect `--prompt-data` pointing to JSONL; convert from this repo’s parquet/jsonl to Slime’s format, or follow Slime’s data format documentation.

Config snippets are in `configs/slime/` (vanilla / grpo_lp / dca) and can be used as reference for `adv_mode`, `beta`, and `gamma`.
