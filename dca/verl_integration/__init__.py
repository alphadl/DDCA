"""
VERL integration: unified advantage / reward interface for VERL to call after patching.

This package does not depend on or import verl; VERL installs this package and calls it from VERL code.
Patch VERL at the place where advantages are computed:

  from dca.verl_integration import compute_advantage, reward_for_verl

  rewards = reward_for_verl(correct, lengths, mode=config.reward_mode, gamma=config.gamma)
  advantages = compute_advantage(rewards, lengths, correct_mask=correct, mode=config.adv_mode, beta=config.beta)
"""

from .advantage_estimators import (
    compute_advantage,
    infer_correct_mask,
)
from .reward_shapers import (
    reward_vanilla,
    reward_coupled_lp,
    reward_for_verl,
)
from .verl_hook import compute_advantage_for_verl

__all__ = [
    "compute_advantage",
    "compute_advantage_for_verl",
    "infer_correct_mask",
    "reward_vanilla",
    "reward_coupled_lp",
    "reward_for_verl",
]
