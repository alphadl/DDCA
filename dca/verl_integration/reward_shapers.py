"""
Reward shaping for VERL: used by reward model / reward function to produce
scalar rewards compatible with each baseline (vanilla, grpo_lp, dca).

Use this in your VERL reward function so that:
- vanilla / dca: return 0/1 (correctness only); length is handled in advantage.
- grpo_lp: return (1 - gamma*length) if correct else 0.
"""

import numpy as np
from typing import Optional


def reward_vanilla(correct: np.ndarray) -> np.ndarray:
    """Binary reward for vanilla GRPO: 1 if correct else 0."""
    return np.where(np.asarray(correct, dtype=bool), 1.0, 0.0)


def reward_coupled_lp(
    correct: np.ndarray,
    lengths: np.ndarray,
    gamma: float = 1e-3,
) -> np.ndarray:
    """Coupled length penalty reward: (1 - gamma*length) if correct else 0. Used for grpo_lp baseline."""
    correct = np.asarray(correct, dtype=bool)
    lengths = np.asarray(lengths, dtype=np.float64)
    return np.where(correct, 1.0 - gamma * lengths, 0.0)


def reward_for_verl(
    correct: np.ndarray,
    lengths: np.ndarray,
    mode: str = "vanilla",
    gamma: float = 1e-3,
) -> np.ndarray:
    """
    Single entry for VERL reward function.

    mode: "vanilla" or "dca" -> return 0/1 (length handled in advantage).
          "grpo_lp"         -> return (1 - gamma*length) if correct else 0.
    """
    if mode in ("vanilla", "dca", "dca_rloo"):
        return reward_vanilla(correct)
    if mode == "grpo_lp":
        return reward_coupled_lp(correct, lengths, gamma=gamma)
    raise ValueError("mode must be one of: vanilla, grpo_lp, dca, dca_rloo")
