"""
Unified advantage estimators for VERL GRPO: vanilla, coupled LP, DCA.

VERL typically computes advantage as (r - mean(r)) / std(r) from scalar rewards.
This module provides a single entry point that supports multiple modes so you can
switch baseline via config (e.g. algorithm.adv_mode=vanilla|grpo_lp|dca).
"""

import numpy as np
from typing import Optional

# Relative import for when used inside repo; optional for when copied into verl
try:
    from ..advantage import (
        advantage_dca_grpo,
        advantage_dca_rloo,
        advantage_vanilla_grpo,
        rewards_coupled_lp,
    )
except ImportError:
    from dca.advantage import (
        advantage_dca_grpo,
        advantage_dca_rloo,
        advantage_vanilla_grpo,
        rewards_coupled_lp,
    )


AdvMode = str  # "vanilla" | "grpo_lp" | "dca" | "dca_rloo"


def infer_correct_mask(rewards: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Infer correct_mask from scalar rewards (e.g. 0/1)."""
    return (np.asarray(rewards, dtype=float) > threshold).astype(bool)


def compute_advantage(
    rewards: np.ndarray,
    lengths: np.ndarray,
    correct_mask: Optional[np.ndarray] = None,
    mode: AdvMode = "vanilla",
    *,
    beta: float = 0.2,
    gamma: float = 1e-3,
    use_rloo: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Single entry point for advantage computation, compatible with VERL's per-group batch.

    Parameters
    ----------
    rewards : np.ndarray, shape (G,) or (B, G)
        Per-response reward. For vanilla/dca, can be 0/1 (correctness only).
        For grpo_lp, should already be (1 - gamma*length) if correct else 0.
    lengths : np.ndarray, shape (G,) or (B, G)
        Token length per response.
    correct_mask : np.ndarray, optional, shape (G,) or (B, G)
        True iff response is correct. If None, inferred from rewards > 0.5.
    mode : str
        "vanilla"  : A = (r - mean(r)) / std(r), no length in reward.
        "grpo_lp"  : r is already coupled (1 - gamma*L) for correct; A = (r - mean(r)) / std(r).
        "dca"      : DCA-GRPO advantage (correctness + conditional length).
        "dca_rloo" : DCA-RLOO advantage.
    beta : float
        Length weight for DCA (ignored for vanilla / grpo_lp).
    gamma : float
        Length penalty for grpo_lp reward (only used if you pass raw 0/1 rewards and mode=grpo_lp;
        then we build r = (1 - gamma*L) for correct, 0 else).
    use_rloo : bool
        If True and mode is dca, use DCA-RLOO; else DCA-GRPO.
    eps : float
        Small constant for std.

    Returns
    -------
    advantages : np.ndarray, same shape as rewards
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if correct_mask is None:
        correct_mask = infer_correct_mask(rewards)
    else:
        correct_mask = np.asarray(correct_mask, dtype=bool)

    if rewards.ndim == 1:
        return _compute_advantage_1d(rewards, lengths, correct_mask, mode, beta=beta, gamma=gamma, use_rloo=use_rloo, eps=eps)
    # Batch of groups (B, G)
    B, G = rewards.shape
    if correct_mask.ndim == 1 and correct_mask.size == B * G:
        correct_mask = correct_mask.reshape(B, G)
    if lengths.ndim == 1 and lengths.size == B * G:
        lengths = lengths.reshape(B, G)
    out = np.zeros_like(rewards)
    for b in range(B):
        out[b] = _compute_advantage_1d(
            rewards[b], lengths[b], correct_mask[b], mode, beta=beta, gamma=gamma, use_rloo=use_rloo, eps=eps
        )
    return out


def _compute_advantage_1d(
    rewards: np.ndarray,
    lengths: np.ndarray,
    correct_mask: np.ndarray,
    mode: AdvMode,
    *,
    beta: float,
    gamma: float,
    use_rloo: bool,
    eps: float,
) -> np.ndarray:
    G = rewards.shape[0]
    correct_mask = np.asarray(correct_mask, dtype=bool).reshape(-1)[:G]
    lengths = lengths.reshape(-1)[:G]

    if mode == "vanilla":
        return advantage_vanilla_grpo(rewards, eps=eps)

    if mode == "grpo_lp":
        # Rewards are expected to be coupled: (1 - gamma*L) if correct else 0.
        # If rewards look like 0/1, we can rebuild coupled here for consistency.
        if np.all(np.isin(rewards, (0.0, 1.0))):
            r_coupled = rewards_coupled_lp(correct_mask, lengths, gamma)
        else:
            r_coupled = rewards
        return advantage_vanilla_grpo(r_coupled, eps=eps)

    if mode in ("dca", "dca_rloo"):
        if use_rloo or mode == "dca_rloo":
            return advantage_dca_rloo(correct_mask, lengths, beta=beta, eps=eps)
        return advantage_dca_grpo(correct_mask, lengths, beta=beta, eps=eps)

    raise ValueError("mode must be one of: vanilla, grpo_lp, dca, dca_rloo")
