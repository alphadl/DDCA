"""Unit tests for AES and pass@K."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dca.metrics import pass_at_k, pass_at_k_multi, aes_score


class TestMetrics(unittest.TestCase):
    def test_pass_at_k_k1(self):
        self.assertEqual(pass_at_k(10, 5, 1), 0.5)
        self.assertEqual(pass_at_k(1, 1, 1), 1.0)
        self.assertEqual(pass_at_k(1, 0, 1), 0.0)

    def test_pass_at_k_k2(self):
        # 1 - C(10-5,2)/C(10,2) = 1 - C(5,2)/C(10,2) = 1 - 10/45
        p = pass_at_k(10, 5, 2)
        self.assertAlmostEqual(p, 1 - 10 / 45, places=9)

    def test_aes_improvement(self):
        # Better accuracy, fewer tokens -> positive AES
        aes = aes_score(0.9, 0.8, 500, 1000)
        self.assertGreater(aes, 0)
        self.assertAlmostEqual(3 * (0.9 - 0.8) / 0.8, aes - (1000 - 500) / 1000, places=6)

    def test_aes_accuracy_drop_penalty(self):
        # p < pb -> negative acc term
        aes = aes_score(0.7, 0.8, 500, 1000)
        self.assertAlmostEqual(-5 * (0.8 - 0.7) / 0.8, aes - (1000 - 500) / 1000, places=6)
