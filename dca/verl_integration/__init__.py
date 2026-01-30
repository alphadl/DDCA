"""
VERL integration: unified advantage estimators and reward shapers for GRPO baselines.

Usage in VERL (after patching or registering this as the advantage module):

  from dca.verl_integration import compute_advantage, reward_for_verl

  # In reward function (per response):
  rewards = reward_for_verl(correct, lengths, mode=config.reward_mode, gamma=config.gamma)

  # In advantage computation (per group):
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
