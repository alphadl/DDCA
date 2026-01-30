# DCA: Decoupled Conditional Advantage for Efficient Reasoning

Implementation and scripts for **DCA (Decoupled Conditional Advantage)** for RL-based reasoning (e.g. GRPO/RLOO). This repo provides the **advantage computation** and **evaluation metrics**; it does **not** include a full RL training loop or depend on any specific framework. We provide adapters for [verl](https://github.com/verl-project/verl) and [THUDM/slime](https://github.com/THUDM/slime) so you can plug DCA into your existing trainer.

---

## Table of Contents

- [Motivation](#motivation)
- [Algorithm Details](#algorithm-details)
- [Quick Start: One-Click Reproduce](#quick-start-one-click-reproduce)
- [How to Reproduce (Full Pipeline)](#how-to-reproduce-full-pipeline)
- [How to Evaluate](#how-to-evaluate)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Paper Setup & License](#paper-setup--license)

---

## Motivation

**Problem.** When training language models for reasoning with RL (e.g. GRPO), the policy often **overthinks**: it produces correct answers but with unnecessarily long reasoning. Adding a **naive length penalty** to the reward (e.g. \(r = \text{correct} - \gamma \cdot \text{length}\)) often leads to **training collapse** or poor accuracy–efficiency trade-offs.

**Why naive length penalty fails.**

1. **Dilution of the length baseline.** In a mixed group (some correct, some wrong), only correct responses get a length term. Wrong responses have no length penalty, so they pull down the group mean. Correct but concise responses are then **relatively penalized** for being "shorter" than a distorted baseline, so the effective signal is wrong.

2. **Parameter inefficacy.** When all responses in a group are correct, the advantage is normalized (e.g. \((r - \bar{r}) / \sigma\)). The length coefficient \(\gamma\) gets absorbed by this normalization, so **you cannot stably control length** even when you try to tune \(\gamma\).

**Idea.** **Decouple correctness and efficiency**: keep a 0/1 correctness reward, and compute a **separate length advantage** only **within the set of correct responses**. That way the length baseline is not diluted by wrong answers, and we can control efficiency via a dedicated coefficient \(\beta\).

---

## Algorithm Details

### DCA-GRPO (group-relative)

For each prompt we have \(G\) responses with binary correctness \(r_i \in \{0,1\}\) and token lengths \(\ell_i\).

1. **Accuracy advantage** (over all \(G\)):  
   \(A^{\text{acc}}_i = (r_i - \bar{r}) / (\sigma_r + \epsilon)\).

2. **Length score** (only over correct set \(\mathcal{S}_c\)):  
   - Z-score within correct responses: \(z_i = (\ell_i - \mu^*_\ell) / (\sigma^*_\ell + \epsilon)\), where \(\mu^*_\ell, \sigma^*_\ell\) are mean and std of \(\{\ell_j : j \in \mathcal{S}_c\}\).  
   - Bounded score: \(s_i = \sigma(z_i)\) (sigmoid). Shorter correct responses get higher \(s_i\).

3. **Length advantage** (zero-sum within \(\mathcal{S}_c\)):  
   \(A^{\text{len}}_i = -(s_i - \bar{s})\) for \(i \in \mathcal{S}_c\), and 0 otherwise. \(\bar{s}\) is the mean of \(s\) over \(\mathcal{S}_c\).

4. **Total advantage:**  
   \(A_i = A^{\text{acc}}_i + \beta \cdot A^{\text{len}}_i\).  
   \(\beta\) (e.g. 0.2) trades off accuracy vs efficiency.

**Implementation:** `dca.advantage.advantage_dca_grpo(correct_mask, lengths, beta)`.

### DCA-RLOO (leave-one-out)

Same length score \(s_i\); accuracy and length advantages use leave-one-out baselines (no normalization). Preferable when group size \(G\) is small.

**Implementation:** `dca.advantage.advantage_dca_rloo(correct_mask, lengths, beta)`.

### Baselines (for comparison)

- **Vanilla GRPO:** \(A_i = (r_i - \bar{r}) / \sigma_r\), \(r \in \{0,1\}\) (no length).
- **Coupled length penalty (GRPO+LP):** \(r_i = (1 - \gamma \ell_i)\) if correct else 0, then \(A_i = (r_i - \bar{r}) / \sigma_r\).  
  Implemented as `rewards_coupled_lp` + vanilla advantage in `dca.verl_integration` / `dca.slime_integration`.

---

## Quick Start: One-Click Reproduce

**1. Clone and install**

```bash
cd efficient_reason_DCA
pip install -r requirements.txt
```

**2. One command** (prepare data → demo inference → evaluate; runs in seconds with built-in samples):

```bash
BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh
```

You should see pass@1, avg_tokens, and num_samples printed. Data and demo results are under `data/processed/`.

**3. Optional environment variables** (all have defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/processed` | Where to write/read data and results |
| `TRAIN_SIZE` / `VAL_SIZE` | 200 / 50 | Max train/val samples (when using HuggingFace) |
| `BUILTIN_ONLY` | 0 | Set to `1` to skip HF and use built-in samples only (quick demo) |
| `USE_MATH` | 0 | Set to `1` to add MATH dataset (~1:2 with GSM8K) when HF is available |
| `RESULTS_FILE` | (empty) | If set, skip prepare/demo and only run evaluation on this JSONL |

---

## How to Reproduce (Full Pipeline)

### Data preparation

Generate small-scale train/val data (parquet for frameworks, JSONL for our evaluator):

```bash
python scripts/prepare_data.py --output_dir data/processed --train_size 500 --val_size 100
```

With built-in samples only (no HuggingFace download):

```bash
python scripts/prepare_data.py --output_dir data/processed --train_size 10 --val_size 5 --builtin_only
```

Outputs: `data/processed/train.parquet`, `val.parquet`, `train.jsonl`, `val.jsonl`, `test_gsm8k.jsonl`.

### Reproduce with VERL

1. Install this repo in your VERL environment: `pip install -e .` (from this repo root).
2. Patch VERL's advantage computation to call our interface: see [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md) for the exact code change (one place: replace `(rewards - mean) / std` with `compute_advantage(..., mode="dca", beta=0.2)`).
3. Point VERL to `data/processed/train.parquet` (or your converted path).
4. Run three baselines:  
   `RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_verl_baselines.sh`  
   (with your VERL overrides as needed).

### Reproduce with Slime (THUDM/slime)

1. Install this repo where Slime can import it; patch Slime's advantage step to call `compute_advantage_for_slime(batch, adv_mode=..., beta=0.2)`. See [docs/INTEGRATION_SLIME.md](docs/INTEGRATION_SLIME.md).
2. Use the same prepared data (convert to Slime's expected format if required).
3. Run: `RUN_BASELINES=vanilla,grpo_lp,dca ./scripts/run_slime_baselines.sh` (with your Slime args). The script passes `ADV_MODE` via environment; your patch reads it (e.g. `os.environ.get("ADV_MODE", "vanilla")`).

### Evaluating existing results (no training)

If you already have a results file (e.g. from VERL or Slime):

```bash
RESULTS_FILE=/path/to/results.jsonl ./scripts/run_full_pipeline.sh
```

This skips data prep and demo inference and only runs evaluation (see [How to Evaluate](#how-to-evaluate)).

---

## How to Evaluate

### Results format

Evaluation expects a **JSONL** file: one JSON object per line, each with at least:

- `predictions`: list of strings (K rollouts per problem) or a single string.
- `lengths`: list of token counts per rollout, or a single integer.
- `ground_truth` or `answer`: reference answer string.

Optional: `question_id` or `index` for logging.

### Run evaluation

```bash
python scripts/evaluate.py --results /path/to/results.jsonl --k 1
```

For pass@k with multiple rollouts per problem, use e.g. `--k 3`.  
To compute **AES (Accuracy–Efficiency Score)** against a baseline run:

```bash
python scripts/evaluate.py --results results_dca.jsonl --base_results results_vanilla.jsonl --k 1
```

### Metrics

| Metric | Description |
|--------|-------------|
| **pass@1** | Fraction of problems with at least one correct rollout (using first rollout per problem). |
| **pass@k** | 1 − C(n−c, k)/C(n, k) averaged over problems; n = rollouts per problem, c = number correct. |
| **avg_tokens** | Mean token count per response (over all rollouts). |
| **AES** | Accuracy–Efficiency Score vs a baseline: combines relative gain in pass@1 and relative reduction in avg_tokens (see `dca.metrics.aes_score`). Requires `--base_results`. |

Math answer equivalence uses normalization (strip, lower, extract `####` or `\boxed{}`, numeric comparison) via `dca.data_utils.is_equivalent_math`.

---

## Project Structure

```
efficient_reason_DCA/
├── dca/
│   ├── advantage.py           # DCA-GRPO, DCA-RLOO, length_score_z_sigmoid, baselines
│   ├── metrics.py             # pass@k, AES, compute_accuracy, compute_avg_tokens
│   ├── data_utils.py          # load GSM8K/MATH, normalize math answers, is_equivalent_math
│   ├── verl_integration/      # compute_advantage, reward_for_verl, compute_advantage_for_verl
│   └── slime_integration/     # compute_advantage_for_slime, reward_for_slime
├── scripts/
│   ├── run_full_pipeline.sh   # One-click: prepare → demo → evaluate
│   ├── prepare_data.py        # Small-scale data (parquet + jsonl)
│   ├── demo_inference.py      # Synthetic results when no VERL/Slime
│   ├── evaluate.py            # CLI: pass@1, pass@k, avg_tokens, AES
│   ├── run_verl_baselines.sh  # Run vanilla / grpo_lp / dca with VERL
│   ├── run_slime_baselines.sh # Run vanilla / grpo_lp / dca with Slime
│   ├── run_verl_comparison.py # Local comparison of advantage modes (no framework)
│   ├── verify_dca.py          # Check formulas (parameter inefficacy, zero-sum length)
│   ├── cpu_mini_validate.py   # Toy policy: DCA vs coupled LP (CPU only)
│   ├── train_dca.py           # Dry-run API check for DCA in a training loop
│   └── run_tests.py           # Run all unit tests
├── configs/
│   ├── experiment.yaml        # Paper-like training/eval config
│   ├── verl/                  # VERL config snippets (vanilla, grpo_lp, dca)
│   └── slime/                 # Slime config snippets
├── docs/
│   ├── INTEGRATION_VERL.md    # How to patch VERL to use DCA
│   └── INTEGRATION_SLIME.md  # How to patch Slime to use DCA
├── tests/
│   ├── test_advantage.py     # DCA formulas, length score, baselines
│   ├── test_metrics.py       # pass@k, AES
│   ├── test_verl_integration.py
│   └── test_slime_integration.py
├── requirements.txt
└── README.md
```

---

## How to Contribute

We welcome bug fixes, clearer docs, and small extensions that stay aligned with the paper.

1. **Fork and clone** the repo; create a branch for your change.
2. **Run tests** before submitting:  
   `python scripts/run_tests.py`  
   (or `pytest tests/ -v` if you use pytest). All tests should pass.
3. **Optional checks:**  
   - Formula sanity: `python scripts/verify_dca.py`  
   - Tiny DCA vs LP: `python scripts/cpu_mini_validate.py`
4. **Code style:** Keep the existing style (e.g. NumPy for arrays, type hints where helpful). No need for a heavy formatter unless the project adds one later.
5. **Open a Pull Request** against `main` with a short description of the change and, if relevant, how you ran evaluation.

If you add a new dependency, add it to `requirements.txt` with a version constraint.

---

## Paper Setup & License

**Experiment setup (from the paper).**  
Models: Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B. Training: AIME + MATH ~1:2, 2500 samples. Evaluation: GSM8K, MATH500, AMC23, AIME25. Hyperparameters: temperature 0.6, top_p 0.95, max_tokens 16384; 3 rollouts per problem for GSM8K/MATH500, 10 for AMC/AIME. Metrics: pass@1, pass@10, avg_tokens, AES. Recommended \(\beta \approx 0.2\) for DCA.

**License.**  
This implementation is for research use. Models and datasets follow their respective licenses.
