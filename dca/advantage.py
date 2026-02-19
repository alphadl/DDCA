"""
Decoupled Conditional Advantage (DCA) / Dynamic DCA (DDCA) for RLVR.

Implements DCA-GRPO and DCA-RLOO (methodology: decoupled accuracy + conditional length advantage).
DDCA (use_dynamic=True): scale length advantage by pass rate ρ = n/G (Difficulty-Aware Coefficient).
  - Hard problems (ρ→0): length term suppressed, focus on accuracy.
  - Easy problems (ρ→1): full length penalty for efficiency.
"""

import numpy as np
from typing import List, Callable, Optional


def is_correct(pred: str, gt: str, equiv_fn: Optional[Callable[[str, str], bool]] = None) -> bool:
    """Check if predicted answer is equivalent to ground truth."""
    if equiv_fn is not None:
        return equiv_fn(pred, gt)
    # Default: normalize and compare (strip, lower, extract last number if needed)
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()
    if pred == gt:
        return True
    # Try extracting numeric answer (e.g. "#### 64" -> "64")
    if "####" in pred:
        pred = pred.split("####")[-1].strip()
    if "####" in gt:
        gt = gt.split("####")[-1].strip()
    return pred == gt


def extract_answer(text: str) -> str:
    """Extract boxed or #### answer from model output."""
    text = str(text).strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    if "\\boxed{" in text:
        start = text.rfind("\\boxed{")
        if start >= 0:
            depth = 0
            for i in range(start + 7, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    if depth == 0:
                        return text[start + 7 : i].strip()
                    depth -= 1
    return text


def length_score_z_sigmoid(lengths: np.ndarray, correct_mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Conditional length score: Z-score within correct set, then sigmoid.
    Only defined for correct responses; others can be 0 or NaN (handled by caller).

    Eq. (12)-(13): z_i = (|o_i| - mu*_len) / (sigma*_len + eps),  s_i = sigmoid(z_i).
    """
    correct_lengths = lengths[correct_mask]
    if correct_lengths.size == 0:
        return np.zeros_like(lengths, dtype=float)

    mu_len = np.mean(correct_lengths)
    sigma_len = np.std(correct_lengths)
    if sigma_len < eps:
        sigma_len = eps

    z = (lengths - mu_len) / (sigma_len + eps)
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    return s


def advantage_dca_grpo(
    correct_mask: np.ndarray,
    lengths: np.ndarray,
    beta: float,
    eps: float = 1e-8,
    use_dynamic: bool = True,
) -> np.ndarray:
    """
    DCA-GRPO (or DDCA-GRPO when use_dynamic=True): decoupled advantages for accuracy and length.

    - Accuracy: A_acc_i = (r_acc_i - mu_acc) / (sigma_acc + eps) over all G.
    - Length:  A_len_i = -(s_i - s_bar) for i in Sc, else 0. s_bar = mean(s) over Sc.
      If use_dynamic (DDCA): A_len_i *= (n/G) so hard problems (low pass rate) get less length penalty.
    - Total:   A_i = A_acc_i + beta * A_len_i.

    correct_mask: bool array [G], True iff response is correct.
    lengths: int array [G], token lengths.
    beta: length penalty coefficient.
    use_dynamic: if True (default), scale length advantage by ρ = n/G (DDCA, Eq.13).
    """
    G = correct_mask.shape[0]
    # Accuracy reward: 1 if correct else 0
    r_acc = correct_mask.astype(np.float64)
    mu_acc = np.mean(r_acc)
    sigma_acc = np.std(r_acc)
    if sigma_acc < eps:
        sigma_acc = eps
    A_acc = (r_acc - mu_acc) / (sigma_acc + eps)

    # Length score only for correct responses
    s = length_score_z_sigmoid(lengths, correct_mask, eps)
    Sc = np.where(correct_mask)[0]
    n = len(Sc)
    A_len = np.zeros(G, dtype=np.float64)
    if n > 0:
        s_correct = s[correct_mask]
        s_bar = np.mean(s_correct)
        A_len[correct_mask] = -(s[correct_mask] - s_bar)
        if use_dynamic:
            # DDCA: scale by pass rate ρ = n/G (Difficulty-Aware Coefficient)
            rho = n / G
            A_len[correct_mask] *= rho

    return A_acc + beta * A_len


def advantage_dca_rloo(
    correct_mask: np.ndarray,
    lengths: np.ndarray,
    beta: float,
    eps: float = 1e-8,
    use_dynamic: bool = True,
) -> np.ndarray:
    """
    DCA-RLOO (or DDCA-RLOO when use_dynamic=True): leave-one-out baseline.

    - Accuracy: A_acc_i = r_acc_i - mean(r_acc over j != i).
    - Length:   A_len_i = -(s_i - mean(s over j in Sc, j != i)) for i in Sc, else 0.
      If use_dynamic (DDCA): A_len_i *= (n/G) (Eq.13).
    """
    G = correct_mask.shape[0]
    r_acc = correct_mask.astype(np.float64)
    A_acc = np.zeros(G, dtype=np.float64)
    for i in range(G):
        others = np.array([j for j in range(G) if j != i])
        A_acc[i] = r_acc[i] - np.mean(r_acc[others])

    s = length_score_z_sigmoid(lengths, correct_mask, eps)
    Sc = np.where(correct_mask)[0]
    n = len(Sc)
    A_len = np.zeros(G, dtype=np.float64)
    for i in range(G):
        if not correct_mask[i]:
            continue
        others_in_Sc = [j for j in Sc if j != i]
        if len(others_in_Sc) == 0:
            A_len[i] = 0.0
        else:
            s_bar_i = np.mean(s[others_in_Sc])
            A_len[i] = -(s[i] - s_bar_i)
    if use_dynamic and n > 0:
        rho = n / G
        A_len[correct_mask] *= rho

    return A_acc + beta * A_len


def advantage_vanilla_grpo(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(r - mean(r)) / (std(r) + eps)."""
    mu = np.mean(rewards)
    sigma = np.std(rewards)
    if sigma < eps:
        sigma = eps
    return (rewards - mu) / (sigma + eps)


def rewards_coupled_lp(
    correct_mask: np.ndarray,
    lengths: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Coupled reward with length penalty: r = (1 - gamma*|o|) if correct else 0.
    Used for baseline GRPO+LP.
    """
    r = np.zeros(len(correct_mask), dtype=np.float64)
    r[correct_mask] = 1.0 - gamma * lengths[correct_mask]
    return r
