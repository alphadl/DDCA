#!/usr/bin/env bash
# Run Slime GRPO baselines: vanilla, grpo_lp, dca.
# For use with THUDM/slime: https://github.com/THUDM/slime
#
# Prerequisites: Slime and this package installed (pip install -e .); Slime patched
#   as in docs/INTEGRATION_SLIME.md so advantage computation calls dca.slime_integration
#   (compute_advantage_for_slime / reward_for_slime).
#
# Usage:
#   From Slime repo root, or set SLIME_REPO, then:
#   RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_slime_baselines.sh [Slime args...]
#   SLIME_CMD defaults to: python train.py (if Slime uses bash scripts/run-xxx.sh, set SLIME_CMD="bash scripts/run-xxx.sh")
#
# Optional: run a subset, e.g. RUN_BASELINES=vanilla,dca ./scripts/run_slime_baselines.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# THUDM/slime entry is typically python train.py (run from Slime repo root)
SLIME_CMD="${SLIME_CMD:-python train.py}"
# adv_mode is passed via env; Slime patch can use os.environ.get("ADV_MODE", "vanilla")
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
