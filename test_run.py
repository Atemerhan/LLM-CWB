"""Unit tests for the core benchmark engine (run.py).

Task under test: next-move prediction with BFS-distance-field ground truth,
randomized goal placement, difficulty stratification, and JSON persistence.
Run with: python -m unittest test_run
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest

import run
from run import OPEN, WALL


class DistanceFieldTests(unittest.TestCase):
    def test_goal_is_zero(self) -> None:
        self.assertEqual(run.distance_field([[OPEN]], (0, 0))[0][0], 0)

    def test_open_grid_manhattan(self) -> None:
        grid = [[OPEN] * 3 for _ in range(3)]
        dist = run.distance_field(grid, (2, 2))
        self.assertEqual(dist[0][0], 4)
        self.assertEqual(dist[0][2], 2)

    def test_wall_lengthens_distance(self) -> None:
        grid = [
            [OPEN, WALL, OPEN],
            [OPEN, WALL, OPEN],
            [OPEN, OPEN, OPEN],
        ]
        self.assertEqual(run.distance_field(grid, (0, 2))[0][0], 6)

    def test_unreachable_is_none(self) -> None:
        grid = [
            [OPEN, OPEN, WALL],
            [OPEN, WALL, WALL],
            [WALL, WALL, OPEN],
        ]
        self.assertIsNone(run.distance_field(grid, (2, 2))[0][0])

    def test_walls_are_none(self) -> None:
        grid = [[OPEN, WALL], [OPEN, OPEN]]
        self.assertIsNone(run.distance_field(grid, (1, 1))[0][1])

    def test_walled_goal_all_none(self) -> None:
        grid = [[OPEN, OPEN], [OPEN, WALL]]
        dist = run.distance_field(grid, (1, 1))
        self.assertTrue(all(d is None for row in dist for d in row))

    def test_empty_grid_raises(self) -> None:
        with self.assertRaises(ValueError):
            run.distance_field([], (0, 0))

    def test_goal_out_of_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            run.distance_field([[OPEN]], (5, 5))


class CorrectMovesTests(unittest.TestCase):
    def test_single_correct_move_down(self) -> None:
        grid = [
            [OPEN, WALL],
            [OPEN, WALL],
            [OPEN, OPEN],
        ]
        self.assertEqual(run.correct_moves(grid, (0, 0), (2, 0)), ["down"])

    def test_multiple_correct_moves(self) -> None:
        grid = [[OPEN] * 3 for _ in range(3)]
        self.assertEqual(run.correct_moves(grid, (0, 0), (2, 2)), ["down", "right"])

    def test_move_into_wall_is_wrong(self) -> None:
        grid = [[OPEN, WALL], [OPEN, OPEN]]
        self.assertEqual(run.correct_moves(grid, (0, 0), (1, 1)), ["down"])

    def test_off_grid_is_wrong(self) -> None:
        grid = [[OPEN] * 3 for _ in range(3)]
        moves = run.correct_moves(grid, (0, 0), (2, 2))
        self.assertNotIn("up", moves)
        self.assertNotIn("left", moves)

    def test_equal_or_increasing_distance_is_wrong(self) -> None:
        grid = [[OPEN] * 3 for _ in range(3)]
        moves = run.correct_moves(grid, (1, 1), (0, 2))  # goal up-right
        self.assertCountEqual(moves, ["up", "right"])
        self.assertNotIn("down", moves)
        self.assertNotIn("left", moves)

    def test_unreachable_position_has_no_moves(self) -> None:
        grid = [[OPEN, WALL], [WALL, OPEN]]
        self.assertEqual(run.correct_moves(grid, (0, 0), (1, 1)), [])


class ClassifyDifficultyTests(unittest.TestCase):
    def test_easy_short_straight(self) -> None:
        # distance == manhattan (ratio 1.0) and short.
        self.assertEqual(run.classify_difficulty(2, 2), "easy")

    def test_hard_high_detour(self) -> None:
        # ratio 2.0 >= 1.5
        self.assertEqual(run.classify_difficulty(8, 4), "hard")

    def test_hard_long_distance(self) -> None:
        # ratio 1.0 but distance >= 10
        self.assertEqual(run.classify_difficulty(12, 12), "hard")

    def test_medium_in_between(self) -> None:
        # ratio ~1.2, distance 6 -> neither easy nor hard
        self.assertEqual(run.classify_difficulty(6, 5), "medium")

    def test_straight_but_not_short_is_not_easy(self) -> None:
        # ratio 1.0 but distance 5 (> EASY_MAX_DISTANCE) -> medium
        self.assertEqual(run.classify_difficulty(5, 5), "medium")

    def test_zero_manhattan_raises(self) -> None:
        with self.assertRaises(ValueError):
            run.classify_difficulty(0, 0)


class GenerateGridTests(unittest.TestCase):
    def test_dimensions(self) -> None:
        grid = run.generate_grid(4, 5, wall_prob=0.3, rng=random.Random(0))
        self.assertEqual(len(grid), 4)
        self.assertTrue(all(len(row) == 5 for row in grid))

    def test_all_open_when_wall_prob_zero(self) -> None:
        grid = run.generate_grid(3, 3, wall_prob=0.0, rng=random.Random(0))
        self.assertTrue(all(cell == OPEN for row in grid for cell in row))

    def test_reproducible_with_seed(self) -> None:
        g1 = run.generate_grid(6, 6, 0.4, random.Random(123))
        g2 = run.generate_grid(6, 6, 0.4, random.Random(123))
        self.assertEqual(g1, g2)

    def test_invalid_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            run.generate_grid(0, 3)

    def test_invalid_wall_prob_raises(self) -> None:
        with self.assertRaises(ValueError):
            run.generate_grid(3, 3, wall_prob=1.5)


class MakeSampleTests(unittest.TestCase):
    def test_position_reachable_and_not_goal(self) -> None:
        rng = random.Random(7)
        for _ in range(50):
            s = run.make_sample(8, 8, wall_prob=0.25, rng=rng)
            self.assertNotEqual(s.position, s.goal)
            self.assertIsNotNone(run.cell_distance(s.grid, s.position, s.goal))

    def test_goal_is_randomized_not_corner(self) -> None:
        rng = random.Random(1)
        goals = {run.make_sample(8, 8, rng=rng).goal for _ in range(40)}
        # Should land on many distinct cells, not just the bottom-right corner.
        self.assertGreater(len(goals), 5)
        self.assertNotEqual(goals, {(7, 7)})

    def test_correct_moves_non_empty(self) -> None:
        rng = random.Random(3)
        for _ in range(50):
            s = run.make_sample(8, 8, wall_prob=0.25, rng=rng)
            self.assertTrue(s.correct_moves)

    def test_label_matches_oracle(self) -> None:
        rng = random.Random(9)
        for _ in range(50):
            s = run.make_sample(8, 8, wall_prob=0.25, rng=rng)
            self.assertEqual(
                s.correct_moves, run.correct_moves(s.grid, s.position, s.goal)
            )

    def test_metadata_consistent(self) -> None:
        rng = random.Random(4)
        for _ in range(50):
            s = run.make_sample(8, 8, rng=rng)
            pr, pc = s.position
            gr, gc = s.goal
            self.assertEqual(s.manhattan, abs(pr - gr) + abs(pc - gc))
            self.assertGreaterEqual(s.distance, s.manhattan)  # walls only add
            self.assertAlmostEqual(s.detour_ratio, round(s.distance / s.manhattan, 3))
            self.assertEqual(
                s.difficulty, run.classify_difficulty(s.distance, s.manhattan)
            )

    def test_requested_difficulty_is_respected(self) -> None:
        rng = random.Random(5)
        for difficulty in run.DIFFICULTIES:
            s = run.make_sample(8, 8, rng=rng, difficulty=difficulty)
            self.assertEqual(s.difficulty, difficulty)

    def test_unknown_difficulty_raises(self) -> None:
        with self.assertRaises(ValueError):
            run.make_sample(8, 8, difficulty="impossible")

    def test_exhausts_attempts(self) -> None:
        # wall_prob 1.0 -> no open cells, never succeeds.
        with self.assertRaises(RuntimeError):
            run.make_sample(6, 6, wall_prob=1.0, rng=random.Random(0),
                            max_attempts=30)


class DatasetTests(unittest.TestCase):
    def test_size(self) -> None:
        self.assertEqual(len(run.generate_dataset(9, 8, 8, seed=0)), 9)

    def test_stratified_balanced(self) -> None:
        ds = run.generate_dataset(9, 8, 8, seed=1, stratified=True)
        counts = {d: 0 for d in run.DIFFICULTIES}
        for s in ds:
            counts[s.difficulty] += 1
        self.assertEqual(counts, {"easy": 3, "medium": 3, "hard": 3})

    def test_reproducible_with_seed(self) -> None:
        a = run.generate_dataset(9, 8, 8, seed=99)
        b = run.generate_dataset(9, 8, 8, seed=99)
        self.assertEqual([s.position for s in a], [s.position for s in b])
        self.assertEqual([s.goal for s in a], [s.goal for s in b])
        self.assertEqual([s.correct_moves for s in a], [s.correct_moves for s in b])

    def test_labels_consistent_with_oracle(self) -> None:
        for s in run.generate_dataset(12, 8, 8, seed=5):
            self.assertEqual(
                s.correct_moves, run.correct_moves(s.grid, s.position, s.goal)
            )
            self.assertTrue(s.correct_moves)


class SplitCountsTests(unittest.TestCase):
    def test_even_split(self) -> None:
        self.assertEqual(run._split_counts(9, 3), [3, 3, 3])

    def test_remainder_to_front(self) -> None:
        self.assertEqual(run._split_counts(10, 3), [4, 3, 3])


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_samples(self) -> None:
        ds = run.generate_dataset(9, 8, 8, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ds.json")
            run.save_dataset(path, ds, meta={"seed": 7})
            loaded = run.load_dataset(path)
        self.assertEqual(len(loaded), len(ds))
        for a, b in zip(ds, loaded):
            self.assertEqual(a, b)  # Sample is a frozen dataclass

    def test_coords_round_trip_as_tuples(self) -> None:
        ds = run.generate_dataset(3, 8, 8, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ds.json")
            run.save_dataset(path, ds)
            loaded = run.load_dataset(path)
        self.assertIsInstance(loaded[0].position, tuple)
        self.assertIsInstance(loaded[0].goal, tuple)

    def test_bad_version_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ds.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"version": 999, "samples": []}')
            with self.assertRaises(ValueError):
                run.load_dataset(path)


class RenderTests(unittest.TestCase):
    def test_render_marks_agent_and_goal(self) -> None:
        s = run.make_sample(5, 5, rng=random.Random(0))
        text = run.render(s)
        self.assertIn("A", text)
        self.assertIn("G", text)

    def test_render_shape(self) -> None:
        s = run.make_sample(4, 5, rng=random.Random(0))
        lines = run.render(s).splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line) == 5 for line in lines))


if __name__ == "__main__":
    unittest.main()
