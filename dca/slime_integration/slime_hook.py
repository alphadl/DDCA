"""
Slime hook: given a batch dict shaped like THUDM/slime rollout + rewards, compute advantages.

Official Slime repo: https://github.com/THUDM/slime
本包不依赖、不 import slime；集成方向是「Slime 安装本库后，在 Slime 代码里 import 并调用本包」.
Batch 键名可通过参数配置；Slime Sample 使用 response_length（单条），转成 batch 后可能为 response_lengths 或 response_length，用 length_key 指定.
"""

import numpy as np
from typing import Any, Dict, Optional

from dca.verl_integration.advantage_estimators import compute_advantage


def compute_advantage_for_slime(
    batch: Dict[str, Any],
    *,
    adv_mode: str = "vanilla",
    beta: float = 0.2,
    gamma: float = 1e-3,
    use_rloo: bool = False,
    use_dynamic: bool = True,
    reward_key: str = "rewards",
    length_key: str = "response_lengths",
    correct_key: Optional[str] = None,
    group_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute advantages from a Slime-style batch dict.

    Parameters
    ----------
    batch : dict
        Must contain at least:
        - reward_key (default "rewards"): per-response rewards, shape (N,) with N = B*G.
        - length_key (default "response_lengths"): per-response token lengths, shape (N,).
        Optional: correct_key for binary correctness; else inferred from rewards > 0.5.
    adv_mode : str
        "vanilla" | "grpo_lp" | "dca" | "dca_rloo"
    beta, gamma, use_rloo, use_dynamic
        Passed to compute_advantage. use_dynamic=True (DDCA) scales length advantage by ρ=n/G.
    group_size : int, optional
        G (responses per prompt). If None, treat N as one group.

    Returns
    -------
    advantages : np.ndarray, shape (N,) or (B, G)
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
        use_dynamic=use_dynamic,
    )

    if flat and group_size is not None:
        adv = adv.ravel()
    return adv
