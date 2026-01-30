# DCA: Decoupled Conditional Advantage for Efficient Reasoning

Implementation and scripts for DCA. This repo **does not** include a full RL framework and **does not depend on or import verl/slime**; it provides DCA advantage computation. **main**: use with [verl](https://github.com/verl-project/verl) via [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md). **slime_backbone** branch: use with [THUDM/slime](https://github.com/THUDM/slime) via [docs/INTEGRATION_SLIME.md](docs/INTEGRATION_SLIME.md), one-click `./scripts/run_slime_baselines.sh`.

---

## One-Click Reproduce Paper Results

**Setup** (once):

```bash
cd efficient_reason_DCA
pip install -r requirements.txt
```

**One command** (prepare data → demo inference → evaluate, ~seconds):

```bash
BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh
```

Output: pass@1, avg_tokens, etc.; data and results under `data/processed/`.

**Optional env vars** (defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/processed` | Data and results directory |
| `TRAIN_SIZE` / `VAL_SIZE` | 200 / 50 | Sample counts (when HF available) |
| `BUILTIN_ONLY` | 0 | 1 = built-in samples only, no download (quick demo) |
| `USE_MATH` | 0 | 1 = add MATH (~1:2) |
| `RESULTS_FILE` | (empty) | If set, skip prepare/demo and run evaluation only |

**Reproduce with VERL**: Run `python scripts/prepare_data.py --output_dir data/processed [--builtin_only]` to get `train.parquet`, then patch VERL per [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md) and run `./scripts/run_verl_baselines.sh`. **Slime (slime_backbone branch)**: patch [THUDM/slime](https://github.com/THUDM/slime) per [docs/INTEGRATION_SLIME.md](docs/INTEGRATION_SLIME.md) and run `./scripts/run_slime_baselines.sh`. With existing results: `RESULTS_FILE=/path/to/results.jsonl ./scripts/run_full_pipeline.sh` runs evaluation only.

---

## Method (short)

- **Problem**: GRPO-style training tends to overthink; naive length penalty in reward often collapses.
- **Cause**: Wrong answers dilute the length baseline; all-correct groups get normalized so length coefficient is ineffective.
- **Method**: DCA—decouple correctness and efficiency; compute length advantage only within correct set; map length Z-score via Sigmoid to a bounded score.

---

## Project Structure

```
efficient_reason_DCA/
├── dca/                 # Advantage, metrics, data utils; verl_integration / slime_integration (branch slime_backbone)
├── scripts/
│   ├── run_full_pipeline.sh   # One-click: prepare → demo → evaluate
│   ├── prepare_data.py        # Small-scale data (parquet + jsonl)
│   ├── demo_inference.py      # Demo inference (when no VERL/Slime)
│   ├── evaluate.py            # pass@1, avg_tokens, AES
│   ├── run_verl_baselines.sh  # VERL three-baseline comparison
│   ├── run_slime_baselines.sh # Slime three-baseline comparison (slime_backbone branch)
│   ├── run_verl_comparison.py # Local comparison of three advantages
│   ├── verify_dca.py          # Formula verification
│   ├── cpu_mini_validate.py   # Tiny DCA vs coupled LP
│   └── train_dca.py           # DCA API check (dry_run)
├── configs/experiment.yaml   # Experiment config
├── configs/verl/              # VERL config snippets
├── configs/slime/             # Slime config snippets (slime_backbone branch)
└── tests/
```

---

## More

- **Formulas**: `python scripts/verify_dca.py`
- **Tiny validation (CPU)**: `python scripts/cpu_mini_validate.py`
- **Evaluate**: `python scripts/evaluate.py --results path/to/results.jsonl --k 1`
- **Tests**: `python scripts/run_tests.py`

**Paper setup**: Qwen3-1.7B / DeepSeek-R1-Distill-Qwen-1.5B; AIME+MATH ~1:2, 2500 samples; eval on GSM8K/MATH500/AMC/AIME; pass@1, avg_tokens, AES; β≈0.2.

---

## License

Research use; models and data follow their respective licenses.
