"""Unit tests for the response protocol, malformed policy, and scoring.

Run with: python -m unittest test_scoring
"""

from __future__ import annotations

import json
import unittest

import scoring
from run import Sample


def _sample(correct_moves: list[str]) -> Sample:
    """Minimal sample carrying only what scoring inspects."""
    return Sample(
        grid=[[0, 0], [0, 0]],
        position=(0, 0),
        goal=(1, 1),
        correct_moves=correct_moves,
        distance=2,
        manhattan=2,
        detour_ratio=1.0,
        difficulty="easy",
    )


class ParseResponseTests(unittest.TestCase):
    def test_clean_json(self) -> None:
        p = scoring.parse_response('{"move": "up", "confidence": 73}')
        self.assertFalse(p.malformed)
        self.assertEqual(p.move, "up")
        self.assertEqual(p.confidence, 73.0)

    def test_case_and_whitespace_normalized(self) -> None:
        p = scoring.parse_response('{"move": "  RIGHT ", "confidence": 50}')
        self.assertEqual(p.move, "right")

    def test_json_embedded_in_prose(self) -> None:
        text = 'Sure! My answer is {"move": "down", "confidence": 90}. Done.'
        p = scoring.parse_response(text)
        self.assertFalse(p.malformed)
        self.assertEqual(p.move, "down")

    def test_json_in_code_fence(self) -> None:
        text = '```json\n{"move": "left", "confidence": 12}\n```'
        p = scoring.parse_response(text)
        self.assertFalse(p.malformed)
        self.assertEqual(p.move, "left")

    def test_float_confidence_ok(self) -> None:
        p = scoring.parse_response('{"move": "up", "confidence": 99.5}')
        self.assertEqual(p.confidence, 99.5)

    def test_no_json_is_malformed(self) -> None:
        p = scoring.parse_response("I think you should go up.")
        self.assertTrue(p.malformed)

    def test_missing_key_is_malformed(self) -> None:
        self.assertTrue(scoring.parse_response('{"move": "up"}').malformed)
        self.assertTrue(scoring.parse_response('{"confidence": 50}').malformed)

    def test_invalid_move_is_malformed(self) -> None:
        self.assertTrue(
            scoring.parse_response('{"move": "north", "confidence": 50}').malformed
        )

    def test_non_string_move_is_malformed(self) -> None:
        self.assertTrue(
            scoring.parse_response('{"move": 1, "confidence": 50}').malformed
        )

    def test_confidence_out_of_range_is_malformed(self) -> None:
        self.assertTrue(
            scoring.parse_response('{"move": "up", "confidence": 150}').malformed
        )
        self.assertTrue(
            scoring.parse_response('{"move": "up", "confidence": -1}').malformed
        )

    def test_boolean_confidence_is_malformed(self) -> None:
        self.assertTrue(
            scoring.parse_response('{"move": "up", "confidence": true}').malformed
        )

    def test_non_numeric_confidence_is_malformed(self) -> None:
        self.assertTrue(
            scoring.parse_response('{"move": "up", "confidence": "high"}').malformed
        )

    def test_malformed_carries_raw_and_error(self) -> None:
        p = scoring.parse_response("garbage")
        self.assertEqual(p.raw, "garbage")
        self.assertIsNotNone(p.error)


class ScoreTests(unittest.TestCase):
    def test_correct_confident(self) -> None:
        s = _sample(["up", "right"])
        p = scoring.parse_response('{"move": "up", "confidence": 95}')
        result = scoring.score(s, p)
        self.assertTrue(result.correct)
        self.assertTrue(result.confident)
        self.assertFalse(result.confidently_wrong)

    def test_confidently_wrong(self) -> None:
        s = _sample(["up"])
        p = scoring.parse_response('{"move": "down", "confidence": 95}')
        result = scoring.score(s, p)
        self.assertFalse(result.correct)
        self.assertTrue(result.confident)
        self.assertTrue(result.confidently_wrong)

    def test_wrong_but_not_confident(self) -> None:
        s = _sample(["up"])
        p = scoring.parse_response('{"move": "down", "confidence": 10}')
        result = scoring.score(s, p)
        self.assertFalse(result.correct)
        self.assertFalse(result.confident)
        self.assertFalse(result.confidently_wrong)

    def test_threshold_boundary_is_confident(self) -> None:
        s = _sample(["up"])
        p = scoring.parse_response('{"move": "up", "confidence": 80}')
        self.assertTrue(scoring.score(s, p).confident)

    def test_malformed_counts_incorrect_not_confidently_wrong(self) -> None:
        s = _sample(["up"])
        p = scoring.parse_response("no json here")
        result = scoring.score(s, p)
        self.assertFalse(result.correct)
        self.assertTrue(result.malformed)
        self.assertFalse(result.confidently_wrong)
        self.assertIsNone(result.confidence)

    def test_custom_threshold(self) -> None:
        s = _sample(["up"])
        p = scoring.parse_response('{"move": "up", "confidence": 60}')
        self.assertFalse(scoring.score(s, p, threshold=80).confident)
        self.assertTrue(scoring.score(s, p, threshold=50).confident)


class AggregateTests(unittest.TestCase):
    def _score(self, raw: str, correct_moves: list[str]) -> scoring.Score:
        return scoring.score(_sample(correct_moves), scoring.parse_response(raw))

    def test_empty(self) -> None:
        self.assertEqual(scoring.aggregate([]), {"n": 0})

    def test_accuracy_counts_malformed_as_wrong(self) -> None:
        scores = [
            self._score('{"move": "up", "confidence": 90}', ["up"]),     # correct
            self._score('{"move": "down", "confidence": 90}', ["up"]),   # cw
            self._score("garbage", ["up"]),                              # malformed
        ]
        agg = scoring.aggregate(scores)
        self.assertEqual(agg["n"], 3)
        self.assertAlmostEqual(agg["accuracy"], 1 / 3)
        self.assertAlmostEqual(agg["malformed_rate"], 1 / 3)
        self.assertEqual(agg["confidently_wrong"], 1)

    def test_calibration_excludes_malformed(self) -> None:
        scores = [
            self._score('{"move": "up", "confidence": 80}', ["up"]),
            self._score("garbage", ["up"]),
        ]
        agg = scoring.aggregate(scores)
        self.assertEqual(agg["well_formed"], 1)
        self.assertEqual(agg["mean_confidence"], 80.0)
        # confidently-wrong rate is over well-formed responses only
        self.assertAlmostEqual(agg["confidently_wrong_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
