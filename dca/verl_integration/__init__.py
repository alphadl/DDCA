"""
VERL 接入层：统一 advantage / reward 接口，供 VERL 侧 patch 后调用。

本包不依赖、不 import verl；集成方向是「VERL 安装本库后，在 VERL 代码里 import 并调用本包」。
在 VERL 中计算 advantage 的位置 patch 为：

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
