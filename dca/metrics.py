"""
Evaluation metrics from the paper: AES, pass@K.
"""

import numpy as np
from typing import List, Optional


def _comb(n: int, k: int) -> float:
    """C(n, k) for n >= k >= 0. Compatible with Python < 3.8 (no math.comb)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    out = 1.0
    for i in range(k):
        out = out * (n - i) / (i + 1)
    return out


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    pass@k = 1 - C(n-c, k) / C(n, k).
    n: number of samples, c: number of correct samples, k: number of samples per problem.
    """
    if k > n or n == 0:
        return 0.0
    if k == 1:
        return c / n
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def pass_at_k_multi(
    num_samples: int,
    num_correct: List[int],
    k: int,
) -> float:
    """
    Average pass@k over multiple problems, each with num_samples rollouts.
    num_correct[i] = number of correct in problem i.
    """
    if not num_correct:
        return 0.0
    return np.mean([pass_at_k(num_samples, c, k) for c in num_correct])


def aes_score(
    pass_at_1: float,
    pass_at_1_base: float,
    avg_tokens: float,
    avg_tokens_base: float,
) -> float:
    """
    Accuracy-Efficiency Score (Luo et al., 2025). Eq. (18).

    AES = (Lb - L) / Lb + {
        3 * (p - pb) / pb   if p >= pb,
        -5 * (pb - p) / pb if p < pb.
    }
    """
    if avg_tokens_base <= 0:
        token_term = 0.0
    else:
        token_term = (avg_tokens_base - avg_tokens) / avg_tokens_base

    if pass_at_1_base <= 0:
        acc_term = 0.0
    else:
        if pass_at_1 >= pass_at_1_base:
            acc_term = 3.0 * (pass_at_1 - pass_at_1_base) / pass_at_1_base
        else:
            acc_term = -5.0 * (pass_at_1_base - pass_at_1) / pass_at_1_base

    return token_term + acc_term


def compute_accuracy(preds: List[str], labels: List[str], equiv_fn=None) -> float:
    """Accuracy = fraction of (pred, label) pairs that are equivalent."""
    from .advantage import is_correct
    if equiv_fn is None:
        equiv_fn = is_correct
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, labels) if equiv_fn(p, g))
    return correct / len(preds)


def compute_avg_tokens(lengths: List[int]) -> float:
    """Average token count over samples."""
    if not lengths:
        return 0.0
    return float(np.mean(lengths))
