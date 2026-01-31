# Release Notes

- **DCA/DDCA core:** `advantage_dca_grpo` / `advantage_dca_rloo`, conditional length score (Z-score + sigmoid), coupled-reward baselines. **DDCA** (default `use_dynamic=True`): length advantage scaled by pass rate ρ = n/G (Difficulty-Aware Coefficient).
- **Metrics:** AES, pass@K, avg_tokens.
- **Data and evaluation:** GSM8K/MATH loading, answer equivalence, `scripts/evaluate.py`.
- **Verification:** `scripts/verify_dca.py` (parameter inefficacy, zero-sum property), `tests/` unit tests.
- **One-click reproduce:** `BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh`.
- **VERL integration:** See [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md). Requirements: Python ≥3.7, numpy, pyyaml.
