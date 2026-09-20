"""Unit tests for the trivial greedy baseline (baseline.py).

Run with: python -m unittest test_baseline
"""

from __future__ import annotations

import random
import unittest

import baseline
import run
from run import Sample
from scoring import parse_response, score


def _sample(position, goal, correct_moves) -> Sample:
    return Sample(
        grid=[[0, 0], [0, 0]],
        position=position,
        goal=goal,
        correct_moves=correct_moves,
        distance=1,
        manhattan=1,
        detour_ratio=1.0,
        difficulty="easy",
    )


class GreedyMoveTests(unittest.TestCase):
    def test_moves_down_toward_lower_goal(self) -> None:
        self.assertEqual(baseline.greedy_move(_sample((0, 0), (3, 0), [])), "down")

    def test_moves_up_toward_higher_goal(self) -> None:
        self.assertEqual(baseline.greedy_move(_sample((3, 0), (0, 0), [])), "up")

    def test_moves_right_toward_right_goal(self) -> None:
        self.assertEqual(baseline.greedy_move(_sample((0, 0), (0, 3), [])), "right")

    def test_moves_left_toward_left_goal(self) -> None:
        self.assertEqual(baseline.greedy_move(_sample((0, 3), (0, 0), [])), "left")

    def test_prefers_larger_gap_axis(self) -> None:
        # bigger row gap than col gap -> vertical move
        self.assertEqual(baseline.greedy_move(_sample((0, 0), (5, 1), [])), "down")

    def test_response_is_valid_protocol(self) -> None:
        s = _sample((0, 0), (1, 1), ["down", "right"])
        parsed = parse_response(baseline.greedy_response(s))
        self.assertFalse(parsed.malformed)
        self.assertEqual(parsed.confidence, 100.0)
        self.assertIn(parsed.move, run.DIRECTIONS)


class BenchmarkTests(unittest.TestCase):
    def test_structure(self) -> None:
        samples = run.generate_dataset(9, 8, 8, seed=3)
        results = baseline.benchmark(samples)
        self.assertIn("overall", results)
        self.assertEqual(set(results["by_difficulty"]), set(run.DIFFICULTIES))
        self.assertEqual(results["overall"]["n"], 9)

    def test_no_malformed_from_baseline(self) -> None:
        samples = run.generate_dataset(12, 8, 8, seed=4)
        results = baseline.benchmark(samples)
        self.assertEqual(results["overall"]["malformed_rate"], 0.0)

    def test_easy_accuracy_exceeds_hard(self) -> None:
        # The validity claim: greedy is luckier on easy than on hard samples.
        samples = run.generate_dataset(150, 8, 8, seed=11)
        results = baseline.benchmark(samples)
        easy = results["by_difficulty"]["easy"]["accuracy"]
        hard = results["by_difficulty"]["hard"]["accuracy"]
        self.assertGreater(easy, hard)

    def test_greedy_perfect_on_open_grid_sample(self) -> None:
        # On a wall-free path, greedy toward goal is always a correct move.
        s = _sample((0, 0), (2, 2), ["down", "right"])
        result = score(s, parse_response(baseline.greedy_response(s)))
        self.assertTrue(result.correct)


if __name__ == "__main__":
    unittest.main()
