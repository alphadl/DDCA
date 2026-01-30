#!/usr/bin/env python3
"""
Prepare small dataset for DCA pipeline (paper: AIME + MATH ~1:2, 2500; we use smaller).

Outputs:
  - data_dir/train.parquet, val.parquet (verl format: prompt, reward_model.ground_truth, data_source, ability)
  - data_dir/train.jsonl, val.jsonl, test_gsm8k.jsonl (our format: question, answer) for evaluate.py
Uses HuggingFace datasets if available (openai/gsm8k, lighteval/MATH); else writes minimal built-in samples.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# VERL parquet columns
INSTRUCTION_SUFFIX = " Let's think step by step and output the final answer after \"####\"."


def extract_solution_gsm8k(answer_str):
    """Extract answer after #### for GSM8K."""
    s = str(answer_str).strip()
    if "####" in s:
        s = s.split("####")[-1].strip().replace(",", "")
    return s


def extract_solution_math(solution_str):
    """Extract \\boxed{...} from MATH solution."""
    s = str(solution_str).strip()
    if "\\boxed{" in s:
        start = s.rfind("\\boxed{")
        depth = 0
        for i in range(start + 7, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                if depth == 0:
                    return s[start + 7 : i].strip()
                depth -= 1
    return s


def make_parquet_row(question: str, answer: str, data_source: str, ability: str, idx: int, split: str) -> dict:
    """One row in verl parquet format."""
    prompt_content = question.strip() + INSTRUCTION_SUFFIX
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": answer.strip()},
        "extra_info": {"split": split, "index": idx},
    }


def make_jsonl_row(question: str, answer: str, dataset: str) -> dict:
    return {"question": question, "answer": answer, "dataset": dataset}


def load_gsm8k_hf(train_size: int, val_size: int, seed: int):
    """Load GSM8K from HuggingFace datasets."""
    try:
        import datasets
        ds = datasets.load_dataset("openai/gsm8k", "main")
        train = ds["train"].shuffle(seed=seed)
        test = ds["test"]
        train_data = []
        for i, ex in enumerate(train):
            if i >= train_size:
                break
            q = ex["question"]
            a = extract_solution_gsm8k(ex["answer"])
            train_data.append((q, a))
        val_data = []
        for i, ex in enumerate(test):
            if i >= val_size:
                break
            q = ex["question"]
            a = extract_solution_gsm8k(ex["answer"])
            val_data.append((q, a))
        return train_data, val_data, "gsm8k"
    except Exception as e:
        print("HuggingFace GSM8K not available:", e, file=sys.stderr)
        return None, None, None


def load_math_hf(max_train: int, max_val: int, seed: int):
    """Load MATH subset from HuggingFace (lighteval/MATH)."""
    try:
        import datasets
        ds = datasets.load_dataset("lighteval/MATH", "all", trust_remote_code=True)
        # MATH has 'problem', 'solution'; extract \boxed{}
        train = ds.get("train", ds.get("test", []))
        if hasattr(train, "shuffle"):
            train = train.shuffle(seed=seed)
        train_data, val_data = [], []
        for i, ex in enumerate(train):
            q = ex.get("problem", ex.get("question", ""))
            sol = ex.get("solution", "")
            a = extract_solution_math(sol) or str(sol).strip()[:50]
            if i < max_train:
                train_data.append((q, a))
            elif i < max_train + max_val:
                val_data.append((q, a))
            else:
                break
        return train_data, val_data, "math"
    except Exception as e:
        print("HuggingFace MATH not available:", e, file=sys.stderr)
        return None, None, None


def builtin_samples_gsm8k():
    """Minimal built-in math problems (no download)."""
    return [
        ("Janet has 3 apples. She buys 2 more. How many apples does she have?", "5"),
        ("There are 10 birds. 4 fly away. How many remain?", "6"),
        ("A box has 4 rows of 5 cookies. How many cookies?", "20"),
        ("Lisa has 7 pencils. She gives 3 to Tom. How many left?", "4"),
        ("A store has 24 shirts. They sell 9. How many left?", "15"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare small dataset for DCA pipeline")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--train_size", type=int, default=500, help="Max training samples (paper 2500)")
    parser.add_argument("--val_size", type=int, default=100, help="Validation/test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_math", action="store_true", help="Try to add MATH (1:2 with GSM8K if available)")
    parser.add_argument("--builtin_only", action="store_true", help="Use only built-in samples (no HF download, for fast demo)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Paper: AIME + MATH ~1:2, 2500. We use GSM8K as proxy for small pipeline; optionally MATH.
    train_rows_parquet = []
    train_rows_jsonl = []
    val_rows_parquet = []
    val_rows_jsonl = []
    test_rows_jsonl = []

    # 1) GSM8K
    n_gsm8k_train = args.train_size * 2 // 3 if args.use_math else args.train_size
    n_gsm8k_val = args.val_size
    train_gsm8k, val_gsm8k, _ = (None, None, None) if args.builtin_only else load_gsm8k_hf(n_gsm8k_train, n_gsm8k_val, args.seed)
    if train_gsm8k is not None:
        for i, (q, a) in enumerate(train_gsm8k):
            train_rows_parquet.append(make_parquet_row(q, a, "gsm8k", "math", i, "train"))
            train_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))
        for i, (q, a) in enumerate(val_gsm8k):
            val_rows_parquet.append(make_parquet_row(q, a, "gsm8k", "math", i, "val"))
            val_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))
            test_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))
    else:
        builtin = builtin_samples_gsm8k()
        for i, (q, a) in enumerate(builtin):
            train_rows_parquet.append(make_parquet_row(q, a, "gsm8k", "math", i, "train"))
            train_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))
        for i, (q, a) in enumerate(builtin[: min(2, len(builtin))]):
            val_rows_parquet.append(make_parquet_row(q, a, "gsm8k", "math", i, "val"))
            val_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))
            test_rows_jsonl.append(make_jsonl_row(q, a, "gsm8k"))

    # 2) Optional MATH (1:2 ratio with GSM8K)
    if args.use_math:
        n_math_train = args.train_size // 3
        n_math_val = args.val_size // 2
        train_math, val_math, _ = load_math_hf(n_math_train, n_math_val, args.seed)
        if train_math is not None:
            base = len(train_rows_parquet)
            for i, (q, a) in enumerate(train_math):
                train_rows_parquet.append(make_parquet_row(q, a, "math", "math", base + i, "train"))
                train_rows_jsonl.append(make_jsonl_row(q, a, "math"))
            base = len(val_rows_parquet)
            for i, (q, a) in enumerate(val_math):
                val_rows_parquet.append(make_parquet_row(q, a, "math", "math", base + i, "val"))
                val_rows_jsonl.append(make_jsonl_row(q, a, "math"))
                test_rows_jsonl.append(make_jsonl_row(q, a, "math"))

    # Write parquet (verl)
    try:
        import pandas as pd
        pd.DataFrame(train_rows_parquet).to_parquet(out / "train.parquet", index=False)
        pd.DataFrame(val_rows_parquet).to_parquet(out / "val.parquet", index=False)
        print("Wrote", out / "train.parquet", out / "val.parquet")
    except ImportError:
        # Fallback: write as JSONL for parquet-convert later
        with open(out / "train.jsonl.parquet_fallback", "w") as f:
            for r in train_rows_parquet:
                r["prompt"] = json.dumps(r["prompt"])
                r["reward_model"] = json.dumps(r["reward_model"])
                r["extra_info"] = json.dumps(r["extra_info"])
                f.write(json.dumps(r) + "\n")
        with open(out / "val.jsonl.parquet_fallback", "w") as f:
            for r in val_rows_parquet:
                r["prompt"] = json.dumps(r["prompt"])
                r["reward_model"] = json.dumps(r["reward_model"])
                r["extra_info"] = json.dumps(r["extra_info"])
                f.write(json.dumps(r) + "\n")
        print("Wrote fallback JSONL (install pandas+pyarrow for parquet)")

    # Write jsonl (our eval / data_utils)
    with open(out / "train.jsonl", "w") as f:
        for r in train_rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "val.jsonl", "w") as f:
        for r in val_rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "test_gsm8k.jsonl", "w") as f:
        for r in test_rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", out / "train.jsonl", out / "val.jsonl", out / "test_gsm8k.jsonl")
    print("Train size:", len(train_rows_jsonl), "Val/Test size:", len(test_rows_jsonl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
