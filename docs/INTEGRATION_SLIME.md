# Slime 接入 DCA

**Slime 官方仓库**：[THUDM/slime](https://github.com/THUDM/slime)（LLM post-training framework for RL scaling，Megatron + SGLang）

本仓库**不依赖、不 import slime**；`dca.slime_integration` 是供 Slime **调用**的接口（advantage / reward）。在 [THUDM/slime](https://github.com/THUDM/slime) 中安装本库并按下面方式 patch 后，即可切换 vanilla / grpo_lp / dca。

## 安装

在 Slime 环境中安装本库（或克隆到同一机器，将 `efficient_reason_DCA` 加入 PYTHONPATH）：

```bash
cd /path/to/efficient_reason_DCA && pip install -e .
```

## Slime 侧数据结构（THUDM/slime）

- **Sample**（`slime.utils.types.Sample`）：单条样本，含 `reward`（float 或 dict）、`response_length`（int）。
- **Train batch**：由 `list[list[Sample]]` 转成；batch 中通常有 `rewards`、长度数组（可能叫 `response_length` 或 `response_lengths`，取决于 convert 实现）。本库的 `compute_advantage_for_slime` 支持通过 `reward_key`、`length_key` 指定键名。

## Patch 方式

### 1. Advantage 计算

在 Slime 中**计算 GRPO advantage 的位置**（训练后端里对 rewards 做标准化、得到 advantages 的地方），将：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

改为：

```python
from dca.slime_integration import compute_advantage_for_slime

adv_mode = getattr(args, "adv_mode", None) or "vanilla"
if adv_mode in ("vanilla", "grpo_lp", "dca", "dca_rloo"):
    # batch 需包含 rewards 与长度数组；键名与 Slime 的 train data 一致
    advantages = compute_advantage_for_slime(
        batch,
        adv_mode=adv_mode,
        beta=getattr(args, "beta", 0.2),
        gamma=getattr(args, "gamma", 1e-3),
        use_rloo=(adv_mode == "dca_rloo"),
        reward_key="rewards",
        length_key="response_lengths",  # 或 "response_length"，与 Slime 转成 train data 的 key 一致
        group_size=args.n_samples_per_prompt,
    )
else:
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

若 Slime 通过 `--custom-pg-loss-reducer-function-path` 等扩展点注入自定义逻辑，也可在该扩展中调用 `compute_advantage_for_slime`，再返回或写回 advantages。

### 2. Reward 侧

在 Slime 的 reward 函数（如 `--custom-rm-path` 指定的自定义 RM）中，按 mode 生成标量 reward：

- **vanilla / dca**：0/1（仅正确性），长度在 advantage 里用 DCA 处理。
- **grpo_lp**：(1 - gamma*length) if correct else 0。

可直接使用本库接口：

```python
from dca.slime_integration import reward_for_slime

# 在 custom RM 或 reward 后处理里，对每条 sample：
correct = ...   # 是否答对
length = sample.response_length
reward = reward_for_slime(np.array([correct]), np.array([length]), mode=adv_mode, gamma=gamma)[0]
```

Slime 的 `--custom-reward-post-process-path` 若用于对整组 reward 做后处理，也可在其中调用 `reward_for_slime` 生成 0/1 或耦合 reward，再交给默认或 DCA 的 advantage。

### 3. 命令行 / 配置

在 Slime 启动参数中增加（或通过 config 传入）例如：

- `--adv-mode dca`（或 vanilla / grpo_lp）
- `--beta 0.2`（DCA 时）
- `--gamma 0.001`（grpo_lp 时）

具体参数名以 THUDM/slime 的 `slime.utils.arguments` 为准；若暂无 `adv_mode`，可在 patch 处用环境变量或自定义 config 读取。

## 一键运行三 baseline

```bash
# 本地对比三种 advantage 输出（不装 Slime）
python scripts/run_verl_comparison.py

# 在 Slime 仓库根目录下，已 patch 且配置好数据/模型后，可循环跑三组实验（需自行补全 Slime 所需参数）：
RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_slime_baselines.sh [你的 Slime 参数...]
```

`run_slime_baselines.sh` 默认执行 `SLIME_CMD`（默认 `python train.py`），并依次设置环境变量 `ADV_MODE=vanilla`、`grpo_lp`、`dca` 后启动；在 Slime 的 advantage 计算 patch 中可用 `os.environ.get("ADV_MODE", "vanilla")` 读取，再调用 `compute_advantage_for_slime(..., adv_mode=...)`。若 Slime 使用脚本启动（如 `bash scripts/run-glm4-9B.sh`），可设置：

```bash
export SLIME_CMD="bash scripts/run-glm4-9B.sh"
./scripts/run_slime_baselines.sh
```

并在该脚本内通过环境变量或追加参数传入 `adv_mode`、`beta` 等。

## 数据准备

与主分支一致：`python scripts/prepare_data.py --output_dir data/processed [--builtin_only]` 得到 `train.parquet` 等；Slime 使用 `--prompt-data` 指向 JSONL 等格式，需自行从本库的 parquet/jsonl 转为 Slime 所需格式，或使用 Slime 文档中的数据格式。

配置片段见本库 `configs/slime/`（vanilla / grpo_lp / dca），内容可作为 Slime 侧 `adv_mode`、`beta`、`gamma` 的参考。
