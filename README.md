# DCA: Decoupled Conditional Advantage for Efficient Reasoning

DCA advantage computation and evaluation for RL-based reasoning (GRPO/RLOO). No full RL loop; plug into [verl](https://github.com/verl-project/verl) or [THUDM/slime](https://github.com/THUDM/slime) via [INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md) / [INTEGRATION_SLIME.md](docs/INTEGRATION_SLIME.md).

---

## Motivation

- **Problem:** GRPO-style training overthinks; naive length penalty in reward often collapses.
- **Why it fails:** (1) Wrong answers dilute the length baseline so correct-but-short responses get wrong signal. (2) When all correct, normalization absorbs γ so length is uncontrollable.
- **Idea:** Decouple correctness and efficiency: 0/1 reward + **length advantage only within correct set** (β controls efficiency).

---

## Algorithm (short)

- **DCA-GRPO:** \(A_i = A^{\text{acc}}_i + \beta\, A^{\text{len}}_i\). Accuracy: \((r_i - \bar{r})/\sigma_r\). Length: Z-score of \(\ell\) within correct set → sigmoid → zero-mean over correct set; shorter correct → higher \(s_i\) → \(A^{\text{len}}_i = -(s_i - \bar{s})\). API: `advantage_dca_grpo(correct_mask, lengths, beta)`.
- **DCA-RLOO:** Same \(s_i\); leave-one-out baselines. `advantage_dca_rloo(...)`.
- **Baselines:** Vanilla \((r-\bar{r})/\sigma\); coupled LP \(r=(1-\gamma\ell)\) if correct else 0. In `dca.verl_integration` / `dca.slime_integration`.

---

## Quick Start

```bash
pip install -r requirements.txt
BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh
```

Env (optional): `DATA_DIR`, `TRAIN_SIZE`/`VAL_SIZE`, `BUILTIN_ONLY` (1=no HF), `USE_MATH`, `RESULTS_FILE` (skip to eval only).

---

## Reproduce

**Data:** `python scripts/prepare_data.py --output_dir data/processed [--builtin_only]` → `train.parquet`, `val.jsonl`, etc.

**VERL / Slime:** Install this repo, patch advantage step per doc above, point to data, run `./scripts/run_verl_baselines.sh` or `./scripts/run_slime_baselines.sh` with `RUN_BASELINES=vanilla,grpo_lp,dca`.

**Eval only:** `RESULTS_FILE=/path/to/results.jsonl ./scripts/run_full_pipeline.sh`.

---

## Evaluate

**Format:** JSONL with `predictions` (list or str), `lengths` (list or int), `ground_truth`.

```bash
python scripts/evaluate.py --results path/to/results.jsonl --k 1
python scripts/evaluate.py --results dca.jsonl --base_results vanilla.jsonl --k 1   # AES
```

**Metrics:** pass@1, pass@k, avg_tokens, AES (vs baseline). Math equivalence: `is_equivalent_math` in `dca.data_utils`.

---

## Project Structure

```
dca/           advantage.py, metrics.py, data_utils.py, verl_integration/, slime_integration/
scripts/       run_full_pipeline.sh, prepare_data.py, demo_inference.py, evaluate.py,
               run_verl_baselines.sh, run_slime_baselines.sh, verify_dca.py, run_tests.py
configs/       experiment.yaml, verl/, slime/
docs/          INTEGRATION_VERL.md, INTEGRATION_SLIME.md
tests/         test_advantage.py, test_metrics.py, test_verl_integration.py, test_slime_integration.py
```

---

## Contribute

Fork → branch → run `python scripts/run_tests.py` (and optionally `verify_dca.py`, `cpu_mini_validate.py`) → PR. New deps in `requirements.txt`.

---

## Paper & License

Setup: Qwen3-1.7B / DeepSeek-R1-Distill-Qwen-1.5B; AIME+MATH ~1:2, 2500; GSM8K/MATH500/AMC/AIME; β≈0.2. Research use; models/data follow their licenses.
