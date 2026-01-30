"""
Data loading and answer equivalence for GSM8K, MATH500, AMC23, AIME.
Paper: mixed training set AIME + MATH ~1:2, 2500 samples. Eval on four benchmarks.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def normalize_math_answer(s: str) -> str:
    """Normalize for math answer comparison (numbers, boxed, etc.)."""
    s = str(s).strip().lower()
    # Extract #### or \boxed{}
    if "####" in s:
        s = s.split("####")[-1].strip()
    if "\\boxed{" in s:
        start = s.rfind("\\boxed{")
        if start >= 0:
            depth = 0
            for i in range(start + 7, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    if depth == 0:
                        s = s[start + 7 : i].strip()
                        break
                    depth -= 1
    # Remove commas, $, spaces for numeric compare
    s = re.sub(r"[\s,$]", "", s)
    # Common normalizations
    if s.endswith(".0"):
        s = s[:-2]
    return s


def is_equivalent_math(pred: str, gt: str) -> bool:
    """Math answer equivalence (GSM8K, MATH, AMC, AIME style)."""
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gt)
    if p == g:
        return True
    # Try numeric
    try:
        return float(p) == float(g)
    except (ValueError, TypeError):
        pass
    return False


def load_gsm8k(path: str) -> List[Dict[str, Any]]:
    """Load GSM8K (JSONL or JSON with 'question' and 'answer')."""
    path = Path(path)
    data = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        else:
            data = raw.get("data", raw.get("examples", []))
    out = []
    for item in data:
        q = item.get("question", item.get("problem", ""))
        a = item.get("answer", "")
        if "####" in a:
            a = a.split("####")[-1].strip()
        out.append({"question": q, "answer": a, "dataset": "gsm8k"})
    return out


def load_math(path: str) -> List[Dict[str, Any]]:
    """Load MATH (level, problem, solution with final answer)."""
    path = Path(path)
    data = []
    with open(path) as f:
        raw = json.load(f)
    for item in (raw if isinstance(raw, list) else raw.get("data", [])):
        problem = item.get("problem", item.get("question", ""))
        solution = item.get("solution", "")
        # Extract \boxed{...} from solution
        answer = ""
        if "\\boxed{" in solution:
            start = solution.rfind("\\boxed{")
            depth = 0
            for i in range(start + 7, len(solution)):
                if solution[i] == "{":
                    depth += 1
                elif solution[i] == "}":
                    if depth == 0:
                        answer = solution[start + 7 : i].strip()
                        break
                    depth -= 1
        level = item.get("level", 0)
        out.append({"question": problem, "answer": answer, "level": level, "dataset": "math"})
    return out


def load_jsonl_generic(path: str, question_key: str = "question", answer_key: str = "answer") -> List[Dict[str, Any]]:
    """Generic JSONL for AMC23, AIME, etc."""
    path = Path(path)
    out = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            out.append({
                "question": item.get(question_key, item.get("problem", "")),
                "answer": str(item.get(answer_key, "")).strip(),
                "dataset": path.stem,
            })
    return out


def load_dataset(path: str, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Auto-detect or use dataset name: gsm8k, math, math500, amc23, aime.
    """
    path = Path(path)
    name = (dataset or path.stem).lower()
    if "gsm8k" in name:
        return load_gsm8k(str(path))
    if "math" in name:
        return load_math(str(path))
    return load_jsonl_generic(str(path))
