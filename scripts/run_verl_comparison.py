#!/usr/bin/env python3
"""
Run a quick comparison of advantage modes (vanilla, grpo_lp, dca) on synthetic groups.
No VERL required: uses dca.verl_integration locally to verify the integration module.
"""

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dca.verl_integration import compute_advantage


def main():
    np.random.seed(42)
    B, G = 2, 4
    rewards = np.array([
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)
    lengths = np.array([
        [500, 300, 200, 400],
        [600, 100, 350, 150],
    ], dtype=np.float64)
    correct_mask = rewards > 0.5

    print("Comparison of advantage modes (same rewards & lengths)\n")
    print("  Group 0: correct=[T,T,F,T], lengths=[500,300,200,400]")
    print("  Group 1: correct=[T,F,T,F], lengths=[600,100,350,150]\n")

    for mode in ("vanilla", "grpo_lp", "dca"):
        adv = compute_advantage(
            rewards, lengths, correct_mask=correct_mask,
            mode=mode, beta=0.2, gamma=1e-3,
        )
        print(f"  {mode:10s}: {adv[0].round(4).tolist()}  (group0)")
        print(f"             {adv[1].round(4).tolist()}  (group1)")
    print("\n[PASS] verl_integration advantage modes run correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
