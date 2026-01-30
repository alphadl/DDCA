#!/usr/bin/env python3
"""
Demo inference: from val.jsonl produce a synthetic results.jsonl for evaluate.py.

Used when VERL is not available, so the full pipeline (prepare → train/demo → evaluate)
can still be run. Simulates model outputs: some correct, some wrong, with synthetic
token lengths so that pass@1, avg_tokens, etc. are non-trivial.

Usage:
  python scripts/demo_inference.py --input data/processed/val.jsonl --output data/processed/results_demo.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description="Generate demo results from val.jsonl for evaluation")
    parser.add_argument("--input", required=True, help="Input val.jsonl (question, answer, dataset)")
    parser.add_argument("--output", required=True, help="Output results JSONL for evaluate.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correct_ratio", type=float, default=0.6, help="Fraction of samples to mark correct (demo)")
    parser.add_argument("--min_len", type=int, default=50, help="Min synthetic token length")
    parser.add_argument("--max_len", type=int, default=400, help="Max synthetic token length")
    parser.add_argument("--k_rollouts", type=int, default=1, help="Number of rollouts per problem (pass@k)")
    args = parser.parse_args()

    random.seed(args.seed)
    data = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    out_rows = []
    for i, item in enumerate(data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        # Build k_rollouts predictions and lengths
        preds = []
        lengths = []
        for _ in range(args.k_rollouts):
            if random.random() < args.correct_ratio:
                preds.append(answer)
            else:
                try:
                    n = float(answer.replace(",", "").strip())
                    wrong = str(int(n) + random.choice([-1, 1])) if n == int(n) else "0"
                except (ValueError, TypeError):
                    wrong = "0"
                preds.append(wrong)
            lengths.append(random.randint(args.min_len, args.max_len))

        out_rows.append({
            "index": i,
            "question": question,
            "ground_truth": answer,
            "predictions": preds,
            "lengths": lengths,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", len(out_rows), "demo results to", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
