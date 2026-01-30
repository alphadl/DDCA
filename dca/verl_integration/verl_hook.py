"""
VERL hook: given a batch dict shaped like verl's rollout + rewards, compute advantages.

VERL typically passes to the advantage estimator something like:
  - rewards: (B*G,) or (B, G) per-response rewards
  - response_lengths or lengths: (B*G,) or (B, G) token counts

This hook reshapes and calls compute_advantage so you can plug it into verl's
algorithm worker with minimal changes.
"""

import numpy as np
from typing import Any, Dict, Optional

from .advantage_estimators import compute_advantage


def compute_advantage_for_verl(
    batch: Dict[str, Any],
    *,
    adv_mode: str = "vanilla",
    beta: float = 0.2,
    gamma: float = 1e-3,
    use_rloo: bool = False,
    reward_key: str = "rewards",
    length_key: str = "response_lengths",
    correct_key: Optional[str] = None,
    group_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute advantages from a VERL-style batch dict.

    Parameters
    ----------
    batch : dict
        Must contain at least:
        - reward_key (default "rewards"): per-response rewards, shape (N,) with N = B*G.
        - length_key (default "response_lengths"): per-response token lengths, shape (N,).
        Optional: correct_key (e.g. "correct") for binary correctness; else inferred from rewards.
    adv_mode : str
        "vanilla" | "grpo_lp" | "dca" | "dca_rloo"
    beta, gamma, use_rloo
        Passed to compute_advantage.
    group_size : int, optional
        G (responses per prompt). If None, we assume N is one group (flat).

    Returns
    -------
    advantages : np.ndarray, shape (N,) or (B, G)
        Same flat or grouped shape as rewards.
    """
    rewards = np.asarray(batch[reward_key], dtype=np.float64)
    lengths = np.asarray(batch[length_key], dtype=np.float64)
    if correct_key and correct_key in batch:
        correct_mask = np.asarray(batch[correct_key], dtype=bool)
    else:
        correct_mask = None

    flat = rewards.ndim == 1
    if flat and group_size is not None:
        B = rewards.size // group_size
        rewards = rewards.reshape(B, group_size)
        lengths = lengths.reshape(B, group_size)
        if correct_mask is not None:
            correct_mask = correct_mask.reshape(B, group_size)

    adv = compute_advantage(
        rewards,
        lengths,
        correct_mask=correct_mask,
        mode=adv_mode,
        beta=beta,
        gamma=gamma,
        use_rloo=use_rloo,
    )

    if flat and group_size is not None:
        adv = adv.ravel()
    return adv
