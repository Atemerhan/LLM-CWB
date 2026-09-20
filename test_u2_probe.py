"""Unit tests for the pure-python U2 probe statistics & features (u2_probe.py).

Covers the model-free parts (AUROC, bootstrap CIs, input features) — not the
sklearn probe or the torch extraction, which require heavy deps. Run with:
python -m unittest test_u2_probe
"""
from __future__ import annotations

import unittest

import run
import u2_probe as U


class AurocTests(unittest.TestCase):
    def test_known_value(self) -> None:
        # sklearn roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.75
        self.assertAlmostEqual(U._auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, places=6)

    def test_perfect_and_inverted(self) -> None:
        self.assertEqual(U._auroc([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]), 1.0)
        self.assertEqual(U._auroc([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1]), 0.0)

    def test_all_ties_is_half(self) -> None:
        self.assertEqual(U._auroc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)

    def test_single_class_is_nan(self) -> None:
        v = U._auroc([0.1, 0.2, 0.3], [0, 0, 0])
        self.assertNotEqual(v, v)  # nan


class BootstrapTests(unittest.TestCase):
    def test_boot_ci_bounds(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        lo, hi = U._boot_auroc(scores, labels, n_boot=200, seed=1)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_paired_diff_reproducible_and_ordered(self) -> None:
        a = [0.2, 0.3, 0.7, 0.8, 0.6, 0.1]
        b = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        labels = [0, 0, 1, 1, 1, 0]
        p1 = U._paired_boot_diff(a, b, labels, n_boot=300, seed=7)
        p2 = U._paired_boot_diff(a, b, labels, n_boot=300, seed=7)
        self.assertEqual(p1, p2)                      # fixed seed -> identical
        point, lo, hi = p1
        self.assertTrue(lo <= point <= hi)


class InputFeatureTests(unittest.TestCase):
    def test_shape_and_onehot(self) -> None:
        sample = run.make_sample(8, 8, rng=run.random.Random(3))
        f = U.input_features(sample)
        self.assertEqual(len(f), 8)                   # 5 scalars + 3 difficulty one-hot
        self.assertEqual(sum(f[5:]), 1.0)             # exactly one difficulty active
        self.assertTrue(all(isinstance(x, float) for x in f))


if __name__ == "__main__":
    unittest.main()
