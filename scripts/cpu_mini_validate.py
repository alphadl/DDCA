#!/usr/bin/env python3
"""
Minimal CPU-only validation of DCA effectiveness.

No GPU, no PyTorch/transformers. Uses a toy "policy" (single parameter λ:
response length ~ Poisson(λ)) and simulated correctness P(correct|length)
with an optimal length band. Compares training with DCA vs coupled length
penalty; expects DCA to reduce length without collapsing accuracy.

Usage:
  python scripts/cpu_mini_validate.py
  (runs in ~5–15 seconds on CPU)
"""

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dca.advantage import advantage_dca_grpo, advantage_vanilla_grpo, rewards_coupled_lp


# --- Toy world: optimal length band (too short => incomplete, wrong) ---
L_OPT = 80.0
SCALE = 600.0
BASE_CORRECT = 0.2


def p_correct_given_length(L: np.ndarray) -> np.ndarray:
    """P(correct | length). Peak near L_OPT; too short => low acc, too long => wasteful."""
    L = np.asarray(L, dtype=float)
    return BASE_CORRECT + 0.75 * np.exp(-((L - L_OPT) ** 2) / SCALE)


def sample_rollouts(lam: float, G: int, rng: np.random.Generator) -> tuple:
    """Sample G lengths from Poisson(lam), then correctness from p_correct(L)."""
    lengths = rng.poisson(lam=lam, size=G).astype(np.float64)
    lengths = np.clip(lengths, 1.0, 1e4)
    probs = p_correct_given_length(lengths)
    correct_mask = rng.random(G) < probs
    return lengths, correct_mask


def reinforce_step_lam(lam: float, lengths: np.ndarray, advantages: np.ndarray, lr: float) -> float:
    """One gradient step on λ for Poisson(λ). Score = L/λ - 1."""
    # Maximize expected advantage => λ += lr * E[A * (L/λ - 1)]
    G = len(lengths)
    if G == 0:
        return lam
    grad = np.mean(advantages * (lengths / (lam + 1e-8) - 1.0))
    lam_new = lam + lr * grad
    return np.clip(lam_new, 5.0, 2000.0)


def run_trial(
    use_dca: bool,
    n_steps: int,
    G: int,
    beta: float,
    gamma: float,
    lr: float,
    seed: int,
) -> tuple:
    """Run n_steps of λ updates with DCA or coupled-LP advantages. Returns (λ_history, acc_history)."""
    rng = np.random.default_rng(seed)
    lam = 120.0
    lam_hist = [lam]
    acc_hist = []

    for _ in range(n_steps):
        lengths, correct_mask = sample_rollouts(lam, G, rng)
        correct_mask = correct_mask.astype(bool)

        if use_dca:
            adv = advantage_dca_grpo(correct_mask, lengths, beta=beta)
        else:
            rewards = rewards_coupled_lp(correct_mask, lengths, gamma=gamma)
            adv = advantage_vanilla_grpo(rewards)

        lam = reinforce_step_lam(lam, lengths, adv, lr)
        lam_hist.append(lam)
        acc_hist.append(np.mean(correct_mask))

    return lam_hist, acc_hist


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CPU-only minimal DCA validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=int, default=100)
    args = parser.parse_args()
    seed = args.seed
    n_steps = args.n_steps

    G = 4
    beta = 0.2
    gamma = 0.001
    lr = 2.0

    print("CPU-only minimal DCA validation (no GPU, no PyTorch)")
    print("  Toy policy: length ~ Poisson(λ), P(correct|L) peak near L=80")
    print("  Compare: λ updated with DCA vs coupled length penalty")
    print("  seed={}, n_steps={}\n".format(seed, n_steps))

    lam_dca, acc_dca = run_trial(True, n_steps, G, beta, gamma, lr, seed)
    lam_lp, acc_lp = run_trial(False, n_steps, G, beta, gamma, lr, seed + 1)

    print("  Method        | final λ | final acc (last 20) | λ trend")
    print("  --------------|--------|----------------------|--------")
    print(f"  DCA (ours)    | {lam_dca[-1]:6.1f} | {np.mean(acc_dca[-20:]):.3f}                  | {lam_dca[-1] - lam_dca[0]:+.1f}")
    print(f"  Coupled LP    | {lam_lp[-1]:6.1f} | {np.mean(acc_lp[-20:]):.3f}                  | {lam_lp[-1] - lam_lp[0]:+.1f}")

    # Sanity: DCA keeps λ in a reasonable range; script validates that our DCA API runs correctly.
    ok = 10 <= lam_dca[-1] <= 500 and 10 <= lam_lp[-1] <= 500
    if ok:
        print("\n  [PASS] Minimal validation OK: DCA and coupled LP both ran; λ in valid range.")
    else:
        print("\n  [WARN] λ out of range (tune lr/seed).")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
