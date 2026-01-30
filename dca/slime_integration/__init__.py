"""
Slime 接入层：统一 advantage / reward 接口，供 Slime 侧 patch 后调用。

本包不依赖、不 import slime；集成方向是「Slime 安装本库后，在 Slime 代码里 import 并调用本包」.
Advantage 与 reward 逻辑复用 dca.verl_integration（与框架无关），仅 batch 接口按 Slime 约定.
"""

from dca.verl_integration import compute_advantage, reward_for_verl
from .slime_hook import compute_advantage_for_slime

# Slime 侧 reward 与 VERL 相同：0/1 或 (1-gamma*length)，按 mode 选
reward_for_slime = reward_for_verl

__all__ = [
    "compute_advantage",
    "compute_advantage_for_slime",
    "reward_for_slime",
]
