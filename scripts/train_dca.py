#!/usr/bin/env python3
"""
Training script stub for DCA with GRPO/RLOO.

This module computes DCA advantages given (correct_mask, lengths) per prompt.
Integrate with your favorite GRPO/RLOO trainer (e.g., DAPO, custom JAX/PyTorch)
by replacing the advantage computation with dca.advantage_dca_grpo or advantage_dca_rloo.

Usage (conceptual):
  - Load model, ref model, dataset (AIME + MATH ~1:2, 2500 samples).
  - For each batch: sample G responses per prompt, get lengths and correctness.
  - advantages = advantage_dca_grpo(correct_mask, lengths, beta=0.2)
  - Backprop policy gradient with these advantages (e.g., GRPO clipped objective).
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dca import advantage_dca_grpo, advantage_dca_rloo


def main():
    parser = argparse.ArgumentParser(description="DCA training integration")
    parser.add_argument("--algorithm", choices=["grpo", "rloo"], default="grpo")
    parser.add_argument("--beta", type=float, default=0.2, help="Length penalty coefficient")
    parser.add_argument("--dry_run", action="store_true", help="Only check DCA API (no training)")
    args = parser.parse_args()

    if args.dry_run:
        import numpy as np
        G = 4
        correct_mask = np.array([True, True, False, True])
        lengths = np.array([500, 300, 200, 400])
        if args.algorithm == "grpo":
            adv = advantage_dca_grpo(correct_mask, lengths, args.beta)
        else:
            adv = advantage_dca_rloo(correct_mask, lengths, args.beta)
        print("DCA advantage check OK:", adv.shape, adv)
        return 0

    print(
        "For full training, plug DCA into your GRPO/RLOO loop:\n"
        "  from dca import advantage_dca_grpo  # or advantage_dca_rloo\n"
        "  advantages = advantage_dca_grpo(correct_mask, lengths, beta=0.2)\n"
        "Then use these advantages in your policy gradient loss."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
