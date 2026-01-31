"""Unit tests for DCA advantage computation."""

import numpy as np
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dca.advantage import (
    advantage_dca_grpo,
    advantage_dca_rloo,
    advantage_vanilla_grpo,
    rewards_coupled_lp,
    length_score_z_sigmoid,
    is_correct,
    extract_answer,
)


class TestAdvantage(unittest.TestCase):
    def test_length_score_z_sigmoid_bounds(self):
        """Length score s_i should be in (0, 1)."""
        lengths = np.array([100, 200, 300])
        mask = np.ones(3, dtype=bool)
        s = length_score_z_sigmoid(lengths, mask)
        self.assertTrue(np.all(s > 0) and np.all(s < 1))

    def test_dca_grpo_shape(self):
        G = 5
        correct_mask = np.array([True, False, True, True, False])
        lengths = np.array([10, 20, 30, 40, 50])
        adv = advantage_dca_grpo(correct_mask, lengths, beta=0.2)
        self.assertEqual(adv.shape, (G,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_dca_rloo_shape(self):
        G = 4
        correct_mask = np.array([True, True, False, True])
        lengths = np.array([100, 200, 300, 400])
        adv = advantage_dca_rloo(correct_mask, lengths, beta=0.2)
        self.assertEqual(adv.shape, (G,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_dca_length_zero_sum_correct_set(self):
        """Sum of length advantages over correct indices should be 0 (GRPO, use_dynamic=False)."""
        correct_mask = np.array([True, True, False, True])
        lengths = np.array([50, 100, 200, 150])
        s = length_score_z_sigmoid(lengths, correct_mask)
        s_bar = np.mean(s[correct_mask])
        A_len = np.zeros(4)
        A_len[correct_mask] = -(s[correct_mask] - s_bar)
        self.assertLess(np.abs(A_len.sum()), 1e-10)

    def test_ddca_grpo_dynamic_scaling(self):
        """DDCA (use_dynamic=True): length advantage scaled by ρ = n/G."""
        correct_mask = np.array([True, False, True, True])  # n=3, G=4, rho=0.75
        lengths = np.array([50, 100, 150, 200])
        adv_dca = advantage_dca_grpo(correct_mask, lengths, beta=0.2, use_dynamic=False)
        adv_ddca = advantage_dca_grpo(correct_mask, lengths, beta=0.2, use_dynamic=True)
        self.assertEqual(adv_dca.shape, adv_ddca.shape)
        self.assertTrue(np.all(np.isfinite(adv_ddca)))
        # DDCA length term is rho * DCA length term; total differs
        self.assertFalse(np.allclose(adv_dca, adv_ddca))

    def test_ddca_rloo_dynamic_scaling(self):
        """DDCA-RLOO: length advantage scaled by ρ = n/G."""
        correct_mask = np.array([True, True, False])
        lengths = np.array([80, 120, 200])
        adv_ddca = advantage_dca_rloo(correct_mask, lengths, beta=0.2, use_dynamic=True)
        self.assertEqual(adv_ddca.shape, (3,))
        self.assertTrue(np.all(np.isfinite(adv_ddca)))

    def test_coupled_reward_all_wrong(self):
        r = rewards_coupled_lp(np.zeros(4, dtype=bool), np.array([1, 2, 3, 4]), 0.1)
        self.assertTrue(np.all(r == 0))

    def test_extract_answer(self):
        self.assertIn("64", extract_answer("total is 64"))
        self.assertEqual(extract_answer("#### 42"), "42")

    def test_is_correct(self):
        self.assertTrue(is_correct("64", "64"))
        self.assertFalse(is_correct("65", "64"))
