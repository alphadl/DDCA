#!/usr/bin/env bash
# Run VERL GRPO with different baselines: vanilla, grpo_lp, dca.
#
# Prerequisites:
#   1. Install verl and this repo (pip install -e . from repo root).
#   2. Patch verl to use DCA advantage: in the place where advantage is computed,
#      call compute_advantage(rewards, lengths, correct_mask=..., mode=config.algorithm.adv_mode, beta=...)
#      when algorithm.adv_estimator=grpo and algorithm.adv_mode is set.
#   3. Set VERL_BASE or edit BASE_CMD below to your verl training command (one run).
#
# Usage:
#   ./scripts/run_verl_baselines.sh [extra overrides...]
#   e.g. trainer.n_gpus_per_node=8 trainer.total_epochs=5

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Base command: same as verl's grpo example, override experiment_name and adv_mode per run.
# Default: assume verl is installed and use a minimal test run (small batch, 1 epoch) for demo.
VERL_CMD="${VERL_CMD:-python -m verl.trainer.main_ppo}"
BASE_OVERRIDES=(
  algorithm.adv_estimator=grpo
  data.train_batch_size=64
  data.max_prompt_length=512
  data.max_response_length=1024
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.rollout.n=4
  trainer.total_epochs=1
  trainer.experiment_name=grpo_baseline
)

run_one() {
  local name="$1"
  local adv_mode="$2"
  shift 2
  echo "========== Running baseline: $name (adv_mode=$adv_mode) =========="
  $VERL_CMD "${BASE_OVERRIDES[@]}" \
    algorithm.adv_mode="$adv_mode" \
    trainer.experiment_name="verl_${name}" \
    "$@"
}

# Optional: only run a subset via env (e.g. RUN_BASELINES=vanilla,dca)
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

echo "Done. Check trainer.experiment_name (verl_vanilla, verl_grpo_lp, verl_dca) in your logs."
