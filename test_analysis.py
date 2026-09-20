"""Unit tests for calibration analysis (analysis.py).

Run with: python -m unittest test_analysis
"""

from __future__ import annotations

import unittest

import analysis
from scoring import Score


def _score(correct: bool, confidence: float | None, malformed: bool = False) -> Score:
    confident = (confidence is not None) and confidence >= 80
    return Score(
        correct=correct,
        malformed=malformed,
        confident=confident,
        confidently_wrong=confident and not correct and not malformed,
        confidence=confidence,
    )


class CategorizeTests(unittest.TestCase):
    def test_right_high(self) -> None:
        self.assertEqual(analysis.categorize(_score(True, 90)), "right_high")

    def test_right_low(self) -> None:
        self.assertEqual(analysis.categorize(_score(True, 40)), "right_low")

    def test_wrong_high(self) -> None:
        self.assertEqual(analysis.categorize(_score(False, 95)), "wrong_high")

    def test_wrong_low(self) -> None:
        self.assertEqual(analysis.categorize(_score(False, 10)), "wrong_low")

    def test_malformed(self) -> None:
        self.assertEqual(
            analysis.categorize(_score(False, None, malformed=True)), "malformed"
        )

    def test_boundary_is_high(self) -> None:
        self.assertEqual(analysis.categorize(_score(True, 80)), "right_high")

    def test_custom_threshold(self) -> None:
        self.assertEqual(analysis.categorize(_score(True, 60), threshold=50), "right_high")
        self.assertEqual(analysis.categorize(_score(True, 60), threshold=80), "right_low")


class CategoryCountsTests(unittest.TestCase):
    def test_counts_all_categories(self) -> None:
        scores = [
            _score(True, 90),   # right_high
            _score(True, 30),   # right_low
            _score(False, 95),  # wrong_high
            _score(False, 20),  # wrong_low
            _score(False, None, malformed=True),  # malformed
        ]
        counts = analysis.category_counts(scores)
        self.assertEqual(counts["right_high"], 1)
        self.assertEqual(counts["right_low"], 1)
        self.assertEqual(counts["wrong_high"], 1)
        self.assertEqual(counts["wrong_low"], 1)
        self.assertEqual(counts["malformed"], 1)

    def test_counts_sum_to_total(self) -> None:
        scores = [_score(True, 90), _score(False, 10), _score(True, 50)]
        self.assertEqual(sum(analysis.category_counts(scores).values()), 3)


class CalibrationCurveTests(unittest.TestCase):
    def test_bucket_count(self) -> None:
        curve = analysis.calibration_curve([_score(True, 50)], n_buckets=10)
        self.assertEqual(len(curve), 10)

    def test_assignment_and_accuracy(self) -> None:
        # Two responses at 95 (one right, one wrong) land in the last bucket.
        scores = [_score(True, 95), _score(False, 95)]
        curve = analysis.calibration_curve(scores, n_buckets=10)
        last = curve[-1]
        self.assertEqual(last.n, 2)
        self.assertAlmostEqual(last.accuracy, 0.5)
        self.assertAlmostEqual(last.mean_confidence, 95.0)

    def test_confidence_100_in_last_bucket(self) -> None:
        curve = analysis.calibration_curve([_score(True, 100)], n_buckets=10)
        self.assertEqual(curve[-1].n, 1)
        self.assertEqual(sum(b.n for b in curve), 1)

    def test_malformed_excluded(self) -> None:
        scores = [_score(True, 90), _score(False, None, malformed=True)]
        curve = analysis.calibration_curve(scores, n_buckets=10)
        self.assertEqual(sum(b.n for b in curve), 1)

    def test_empty_bucket_is_none(self) -> None:
        curve = analysis.calibration_curve([_score(True, 5)], n_buckets=10)
        self.assertIsNone(curve[-1].accuracy)
        self.assertEqual(curve[0].n, 1)

    def test_invalid_buckets_raise(self) -> None:
        with self.assertRaises(ValueError):
            analysis.calibration_curve([], n_buckets=0)


class ECETests(unittest.TestCase):
    def test_perfect_calibration_is_zero(self) -> None:
        # Confidence 100 and always correct -> gap 0.
        scores = [_score(True, 100) for _ in range(10)]
        self.assertAlmostEqual(analysis.expected_calibration_error(scores), 0.0)

    def test_maximally_miscalibrated(self) -> None:
        # Confidence 100 but always wrong -> gap 1.0.
        scores = [_score(False, 100) for _ in range(10)]
        self.assertAlmostEqual(analysis.expected_calibration_error(scores), 1.0)

    def test_none_when_no_well_formed(self) -> None:
        scores = [_score(False, None, malformed=True)]
        self.assertIsNone(analysis.expected_calibration_error(scores))

    def test_weighted_average(self) -> None:
        # 1 response at 100% correct (gap 0); 1 at 100% wrong (gap 1) -> 0.5.
        scores = [_score(True, 100), _score(False, 100)]
        self.assertAlmostEqual(analysis.expected_calibration_error(scores), 0.5)


if __name__ == "__main__":
    unittest.main()
