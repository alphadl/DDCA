#!/usr/bin/env bash
# 一键运行 Slime GRPO 三 baseline：vanilla、grpo_lp、dca。
# 面向 THUDM/slime：https://github.com/THUDM/slime
#
# 前置：已安装 Slime 与本库（pip install -e .）；并在 Slime 中按 docs/INTEGRATION_SLIME.md patch，
#       使 advantage 计算调用 dca.slime_integration（compute_advantage_for_slime / reward_for_slime）。
#
# 用法：
#   在 Slime 仓库根目录或设置 SLIME_REPO 后执行：
#   RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_slime_baselines.sh [Slime 其余参数...]
#   SLIME_CMD 默认：python train.py（若 Slime 用 bash scripts/run-xxx.sh，则设 SLIME_CMD="bash scripts/run-xxx.sh"）
#
# 可选：只跑部分，如 RUN_BASELINES=vanilla,dca ./scripts/run_slime_baselines.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# THUDM/slime 入口一般为 python train.py（在 Slime 仓库根目录执行）
SLIME_CMD="${SLIME_CMD:-python train.py}"
# 通过环境变量传递 adv_mode，Slime patch 中可用 os.environ.get("ADV_MODE", "vanilla")
run_one() {
  local name="$1"
  local adv_mode="$2"
  shift 2
  echo "========== Running baseline: $name (adv_mode=$adv_mode) =========="
  ADV_MODE="$adv_mode" SLIME_BASELINE_NAME="slime_${name}" $SLIME_CMD "$@"
}

RUN_BASELINES="${RUN_BASELINES:-vanilla,grpo_lp,dca}"
IFS=',' read -ra BASELINES <<< "$RUN_BASELINES"

for b in "${BASELINES[@]}"; do
  case "$b" in
    vanilla)  run_one vanilla vanilla "$@" ;;
    grpo_lp)  run_one grpo_lp grpo_lp "$@" ;;
    dca)      run_one dca dca "$@" ;;
    *)        echo "Unknown baseline: $b" >&2; exit 1 ;;
  esac
done

echo "Done. Check trainer.experiment_name (slime_vanilla, slime_grpo_lp, slime_dca) in your logs."
