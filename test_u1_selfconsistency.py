"""Unit tests for the U1 self-consistency statistics helpers.

Pure stdlib, no network: exercises the AURC / bootstrap / AUROC / rank-correlation
helpers on hand-checked edge cases. Run: python -m unittest test_u1_selfconsistency
"""

import unittest

import u1_selfconsistency as u


class TestU1Stats(unittest.TestCase):

    # (#4) tie-corrected AURC == mean loss on a constant-confidence dataset,
    #      independent of item order. Also covers all-confidence-equal / all-ties.
    def test_constant_confidence_aurc_equals_mean_loss(self):
        for losses in ([0, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0], [1, 1, 1]):
            pairs = [(50.0, l) for l in losses]
            mean_loss = sum(losses) / len(losses)
            self.assertAlmostEqual(u._aurc_value(pairs), mean_loss, places=12)
            val, curve = u.aurc(pairs)
            self.assertAlmostEqual(val, mean_loss, places=12)
            for p in curve:                       # every risk-coverage point is flat
                self.assertAlmostEqual(p["risk"], mean_loss, places=12)

    def test_all_safe(self):
        pairs = [(c, 0) for c in (90, 80, 70, 60)]
        self.assertEqual(u._aurc_value(pairs), 0.0)
        b = u.bootstrap_aurc_vs_baseline(pairs, 200, 1)
        self.assertEqual(b["point"], 0.0)
        self.assertEqual(b["baseline"], 0.0)
        self.assertEqual(b["delta"], 0.0)
        # AUROC undefined with no positives (no unsafe).
        self.assertIsNone(u.auroc_unsafe([0.9, 0.8, 0.7, 0.6], [0, 0, 0, 0]))

    def test_all_unsafe(self):
        pairs = [(c, 1) for c in (90, 80, 70, 60)]
        self.assertEqual(u._aurc_value(pairs), 1.0)
        b = u.bootstrap_aurc_vs_baseline(pairs, 200, 1)
        self.assertEqual(b["point"], 1.0)
        self.assertEqual(b["baseline"], 1.0)
        self.assertEqual(b["delta"], 0.0)
        # AUROC undefined with no negatives (no safe).
        self.assertIsNone(u.auroc_unsafe([0.9, 0.8, 0.7, 0.6], [1, 1, 1, 1]))

    def test_all_ties_rank_corr_undefined(self):
        # constant agreement -> rank correlations undefined (zero variance).
        self.assertIsNone(u.spearman([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]))
        self.assertIsNone(u.kendall_tau_b([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]))

    def test_perfect_signal_beats_baseline(self):
        pairs = [(90, 0), (80, 0), (70, 0), (60, 1), (50, 1), (40, 1)]
        b = u.bootstrap_aurc_vs_baseline(pairs, 1000, 7)
        self.assertLess(b["point"], b["baseline"])       # informative signal
        self.assertLess(b["p_one_sided"], 0.1)           # one-sided bootstrap test
        # AUROC: unsafe concentrated at low confidence -> perfect detector.
        self.assertEqual(u.auroc_unsafe([0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                        [0, 0, 0, 1, 1, 1]), 1.0)
        # Anti-correlated -> AUROC 0.0; rank correlations negative for the good case.
        self.assertEqual(u.auroc_unsafe([0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                        [0, 0, 0, 1, 1, 1]), 0.0)
        self.assertLess(u.spearman([0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                   [0, 0, 0, 1, 1, 1]), 0.0)
        self.assertLess(u.kendall_tau_b([0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                        [0, 0, 0, 1, 1, 1]), 0.0)

    def test_bootstrap_reproducible(self):
        pairs = [(90, 0), (80, 1), (70, 0), (60, 1), (50, 0), (40, 1)]
        a = u.bootstrap_aurc_vs_baseline(pairs, 500, 12345)
        b = u.bootstrap_aurc_vs_baseline(pairs, 500, 12345)
        self.assertEqual(a, b)                           # same seed -> identical
        c = u.bootstrap_aurc_vs_baseline(pairs, 500, 999)
        self.assertNotEqual(a["ci95"], c["ci95"])        # different seed -> differs

    def test_wilson_bounds(self):
        lo, hi = u.wilson(1, 4)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)
        self.assertEqual(u.wilson(0, 0), [None, None])


if __name__ == "__main__":
    unittest.main()
