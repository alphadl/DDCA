"""
Slime integration: unified advantage / reward interface for Slime to call after patching.

This package does not depend on or import slime; Slime installs this package and calls it from Slime code.
Advantage and reward logic are shared with dca.verl_integration (framework-agnostic); only the batch interface follows Slime conventions.
"""

from dca.verl_integration import compute_advantage, reward_for_verl
from .slime_hook import compute_advantage_for_slime

# Slime reward is the same as VERL: 0/1 or (1-gamma*length) depending on mode.
reward_for_slime = reward_for_verl

__all__ = [
    "compute_advantage",
    "compute_advantage_for_slime",
    "reward_for_slime",
]
