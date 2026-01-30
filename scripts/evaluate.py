#!/usr/bin/env python3
"""
Evaluate model outputs: accuracy, avg tokens, pass@K, AES.

Expects a results JSON/JSONL with per-sample fields, e.g.:
  - question_id or index
  - predictions: list of strings (K rollouts) or single string
  - lengths: list of token counts per rollout, or single length
  - ground_truth: str

Usage:
  python scripts/evaluate.py --results results.jsonl --k 3 --base_results base.jsonl
  (base_results optional; if provided, AES is computed using base as Lb, pb.)
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dca.metrics import pass_at_k_multi, aes_score, compute_accuracy, compute_avg_tokens
from dca.data_utils import is_equivalent_math


def load_results(path: str) -> list:
    path = Path(path)
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def evaluate(results: list, k: int = 1) -> dict:
    """
    results: list of {
        "predictions": [str] (K rollouts) or str,
        "lengths": [int] or int,
        "ground_truth": str,
    }
    """
    num_correct_per_problem = []
    all_preds = []
    all_labels = []
    all_lengths = []

    for item in results:
        gt = item.get("ground_truth", item.get("answer", ""))
        preds = item.get("predictions", item.get("prediction", []))
        if isinstance(preds, str):
            preds = [preds]
        lengths = item.get("lengths", item.get("length", [0]))
        if isinstance(lengths, (int, float)):
            lengths = [int(lengths)] * len(preds)
        lengths = list(lengths)[: len(preds)]

        correct_count = sum(1 for p in preds if is_equivalent_math(p, gt))
        num_correct_per_problem.append(correct_count)
        # pass@1 style: use first rollout for accuracy
        all_preds.append(preds[0] if preds else "")
        all_labels.append(gt)
        all_lengths.extend(lengths)

    n_rollouts = 1
    if results:
        preds0 = results[0].get("predictions", [])
        n_rollouts = len(preds0) if isinstance(preds0, list) else 1
    pass_at_1 = compute_accuracy(all_preds, all_labels, equiv_fn=is_equivalent_math)
    pass_at_k_val = pass_at_k_multi(n_rollouts, num_correct_per_problem, min(k, n_rollouts))
    avg_tokens = compute_avg_tokens(all_lengths) if all_lengths else 0.0

    return {
        "pass@1": pass_at_1,
        f"pass@{k}": pass_at_k_val,
        "avg_tokens": avg_tokens,
        "num_samples": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results JSONL path")
    parser.add_argument("--base_results", default=None, help="Baseline results for AES")
    parser.add_argument("--k", type=int, default=1, help="pass@k")
    args = parser.parse_args()

    results = load_results(args.results)
    if not results:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate(results, args.k)
    print("Metrics:", json.dumps(metrics, indent=2))

    if args.base_results:
        base = load_results(args.base_results)
        base_metrics = evaluate(base, args.k)
        aes = aes_score(
            metrics["pass@1"],
            base_metrics["pass@1"],
            metrics["avg_tokens"],
            base_metrics["avg_tokens"],
        )
        print("AES (vs base):", aes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
