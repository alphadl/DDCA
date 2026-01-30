"""Unit tests for dca.verl_integration."""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dca.verl_integration import (
    compute_advantage,
    infer_correct_mask,
    reward_for_verl,
    reward_vanilla,
    reward_coupled_lp,
)


class TestVerlIntegration(unittest.TestCase):
    def test_infer_correct_mask(self):
        r = np.array([1.0, 0.0, 1.0, 0.0])
        m = infer_correct_mask(r)
        self.assertTrue(np.all(m == (r > 0.5)))

    def test_reward_vanilla(self):
        correct = np.array([True, False, True])
        r = reward_vanilla(correct)
        self.assertTrue(np.all(r == [1.0, 0.0, 1.0]))

    def test_reward_coupled_lp(self):
        correct = np.array([True, False, True])
        lengths = np.array([100.0, 200.0, 300.0])
        r = reward_coupled_lp(correct, lengths, gamma=0.001)
        self.assertAlmostEqual(r[0], 0.9)
        self.assertAlmostEqual(r[1], 0.0)
        self.assertAlmostEqual(r[2], 0.7)

    def test_reward_for_verl_modes(self):
        correct = np.array([True, False])
        lengths = np.array([10.0, 20.0])
        rv = reward_for_verl(correct, lengths, mode="vanilla")
        self.assertTrue(np.all(rv == [1.0, 0.0]))
        rl = reward_for_verl(correct, lengths, mode="grpo_lp", gamma=0.01)
        self.assertAlmostEqual(rl[0], 0.9)
        self.assertAlmostEqual(rl[1], 0.0)

    def test_compute_advantage_vanilla_1d(self):
        rewards = np.array([1.0, 1.0, 0.0, 1.0])
        lengths = np.array([100, 200, 300, 400])
        adv = compute_advantage(rewards, lengths, mode="vanilla")
        self.assertEqual(adv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertAlmostEqual(np.mean(adv), 0.0, places=5)

    def test_compute_advantage_dca_1d(self):
        rewards = np.array([1.0, 1.0, 0.0, 1.0])
        lengths = np.array([100, 200, 300, 400])
        adv = compute_advantage(rewards, lengths, mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_compute_advantage_grpo_lp_1d(self):
        rewards = np.array([1.0, 1.0, 0.0, 1.0])
        lengths = np.array([100.0, 200.0, 300.0, 400.0])
        adv = compute_advantage(rewards, lengths, mode="grpo_lp", gamma=0.001)
        self.assertEqual(adv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_compute_advantage_batch_2d(self):
        rewards = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        lengths = np.array([[100, 200, 300], [150, 250, 350]])
        adv = compute_advantage(rewards, lengths, mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_compute_advantage_all_wrong(self):
        rewards = np.array([0.0, 0.0, 0.0])
        lengths = np.array([100, 200, 300])
        adv = compute_advantage(rewards, lengths, mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (3,))
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertAlmostEqual(adv[0], adv[1])
        self.assertAlmostEqual(adv[1], adv[2])

    def test_compute_advantage_all_correct(self):
        rewards = np.array([1.0, 1.0, 1.0])
        lengths = np.array([100, 200, 300])
        adv = compute_advantage(rewards, lengths, mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (3,))
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_compute_advantage_single_correct(self):
        rewards = np.array([0.0, 1.0, 0.0])
        lengths = np.array([100, 200, 300])
        adv = compute_advantage(rewards, lengths, mode="dca", beta=0.2)
        self.assertEqual(adv.shape, (3,))
        self.assertTrue(np.all(np.isfinite(adv)))
