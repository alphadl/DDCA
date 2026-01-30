"""Unit tests for dca.slime_integration. Verify parity with verl_integration for same batch."""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dca.slime_integration import (
    compute_advantage_for_slime,
    reward_for_slime,
    compute_advantage,
)
from dca.verl_integration import compute_advantage_for_verl, reward_for_verl


class TestSlimeIntegration(unittest.TestCase):
    def test_reward_for_slime_equals_reward_for_verl(self):
        correct = np.array([True, False, True])
        lengths = np.array([10.0, 20.0, 30.0])
        r_slime = reward_for_slime(correct, lengths, mode="vanilla")
        r_verl = reward_for_verl(correct, lengths, mode="vanilla")
        np.testing.assert_array_almost_equal(r_slime, r_verl)
        r_slime_lp = reward_for_slime(correct, lengths, mode="grpo_lp", gamma=0.001)
        r_verl_lp = reward_for_verl(correct, lengths, mode="grpo_lp", gamma=0.001)
        np.testing.assert_array_almost_equal(r_slime_lp, r_verl_lp)

    def test_compute_advantage_for_slime_batch_flat(self):
        batch = {
            "rewards": np.array([1.0, 0.0, 1.0, 1.0]),
            "response_lengths": np.array([100, 200, 300, 400]),
        }
        adv = compute_advantage_for_slime(batch, adv_mode="vanilla")
        self.assertEqual(adv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertAlmostEqual(np.mean(adv), 0.0, places=5)

    def test_compute_advantage_for_slime_dca(self):
        batch = {
            "rewards": np.array([1.0, 1.0, 0.0, 1.0]),
            "response_lengths": np.array([100, 200, 300, 400]),
        }
        adv = compute_advantage_for_slime(batch, adv_mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_slime_equals_verl_same_batch(self):
        """Slime hook and VERL hook must give same advantages for same batch."""
        batch = {
            "rewards": np.array([1.0, 0.0, 1.0, 1.0]),
            "response_lengths": np.array([100, 200, 300, 400]),
        }
        for mode in ("vanilla", "grpo_lp", "dca"):
            adv_slime = compute_advantage_for_slime(batch, adv_mode=mode, group_size=4)
            adv_verl = compute_advantage_for_verl(batch, adv_mode=mode, group_size=4)
            np.testing.assert_array_almost_equal(adv_slime, adv_verl, err_msg=f"mode={mode}")

    def test_slime_custom_keys(self):
        """compute_advantage_for_slime accepts custom batch key names."""
        batch = {
            "scores": np.array([1.0, 0.0, 1.0]),
            "lengths": np.array([50, 150, 250]),
        }
        adv = compute_advantage_for_slime(
            batch, adv_mode="dca", beta=0.2,
            reward_key="scores", length_key="lengths",
        )
        self.assertEqual(adv.shape, (3,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_slime_with_group_size_flat(self):
        """Flat (B*G,) batch with group_size reshapes and returns flat."""
        batch = {
            "rewards": np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
            "response_lengths": np.array([100, 200, 300, 400, 500, 600]),
        }
        adv = compute_advantage_for_slime(batch, adv_mode="dca", beta=0.2, group_size=3)
        self.assertEqual(adv.shape, (6,))
        self.assertTrue(np.all(np.isfinite(adv)))
