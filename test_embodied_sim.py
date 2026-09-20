"""Unit tests for the 档 A closed-loop simulation (embodied_sim.py).

Pure Python; reuses the frozen oracle and (where present) the frozen dataset.
Run: python -m unittest test_embodied_sim
"""
from __future__ import annotations

import unittest

import embodied_sim as E
import run


class ClassifyMoveTests(unittest.TestCase):
    def setUp(self) -> None:
        # 2x2: (1,0) is a wall; goal at (0,1).
        self.grid = [[0, 0], [1, 0]]
        self.goal = (0, 1)
        self.dist = run.distance_field(self.grid, self.goal)

    def test_dest(self) -> None:
        self.assertEqual(E.move_dest((0, 0), "right"), (0, 1))
        self.assertEqual(E.move_dest((0, 0), "down"), (1, 0))

    def test_classes(self) -> None:
        g, d = self.grid, self.dist
        self.assertEqual(E.classify_move(g, (0, 0), "right", d), "safe")
        self.assertEqual(E.classify_move(g, (0, 0), "down", d), "wall")
        self.assertEqual(E.classify_move(g, (0, 0), "up", d), "off_grid")
        self.assertEqual(E.classify_move(g, (0, 0), "left", d), "off_grid")
        self.assertEqual(E.classify_move(g, (0, 0), None, d), "malformed")

    def test_safe_moves(self) -> None:
        self.assertEqual(E.safe_moves(self.grid, (0, 0), self.dist), ["right"])


class ArbiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = [[0, 0], [1, 0]]
        self.goal = (0, 1)
        self.dist = run.distance_field(self.grid, self.goal)

    def test_a1_accepts_safe(self) -> None:
        rec = E.arbitrate(E.Authority.A1_SELECT_FROM_SAFE, self.grid, (0, 0),
                          self.goal, "right", self.dist)
        self.assertFalse(rec.vetoed)
        self.assertEqual(rec.executed, "right")
        self.assertEqual(rec.source, "policy")

    def test_a1_vetoes_unsafe_and_falls_back(self) -> None:
        rec = E.arbitrate(E.Authority.A1_SELECT_FROM_SAFE, self.grid, (0, 0),
                          self.goal, "down", self.dist)          # into the wall
        self.assertTrue(rec.vetoed)
        self.assertEqual(rec.source, "fallback")
        self.assertTrue(E.is_safe(self.grid, (0, 0), rec.executed, self.dist))

    def test_ungated_passes_unsafe_through(self) -> None:
        rec = E.arbitrate(E.Authority.UNGATED, self.grid, (0, 0), self.goal,
                          "down", self.dist)
        self.assertFalse(rec.vetoed)
        self.assertEqual(rec.executed, "down")                   # no safety layer


class EpisodeTests(unittest.TestCase):
    def test_oracle_reaches_optimally(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        res = E.run_episode(grid, (0, 0), (2, 2), E.oracle_policy,
                            E.Authority.PLANNER_ONLY)
        self.assertTrue(res.reached)
        self.assertFalse(res.collided)
        self.assertEqual(res.optimal_len, 4)
        self.assertEqual(res.path_len, 4)
        self.assertEqual(res.spl, 1.0)

    def test_spl_bounds_and_success_path(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        res = E.run_episode(grid, (0, 0), (2, 2),
                            E.greedy_manhattan_policy,
                            E.Authority.A1_SELECT_FROM_SAFE)
        self.assertTrue(0.0 <= res.spl <= 1.0)
        if res.reached:
            self.assertGreaterEqual(res.path_len, res.optimal_len)


class FrozenDatasetArmTests(unittest.TestCase):
    """Headline safety-by-construction property on the frozen episodes."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.eps = E.episodes_from_dataset(limit=150)
        except (FileNotFoundError, ValueError):
            cls.eps = None

    def test_a1_never_collides_even_with_adversary(self) -> None:
        if not self.eps:
            self.skipTest("frozen dataset not available")
        s = E.summarize(E.run_arm(self.eps, E.make_random_policy(0),
                                  E.Authority.A1_SELECT_FROM_SAFE))
        self.assertEqual(s["collision_rate"], 0.0)
        self.assertEqual(s["terminations"]["collision"], 0)

    def test_ungated_random_can_collide(self) -> None:
        if not self.eps:
            self.skipTest("frozen dataset not available")
        s = E.summarize(E.run_arm(self.eps, E.make_random_policy(0),
                                  E.Authority.UNGATED))
        self.assertGreater(s["collision_rate"], 0.0)   # harness CAN see collisions

    def test_planner_only_is_safe_and_succeeds(self) -> None:
        if not self.eps:
            self.skipTest("frozen dataset not available")
        s = E.summarize(E.run_arm(self.eps, E.oracle_policy,
                                  E.Authority.PLANNER_ONLY))
        self.assertEqual(s["collision_rate"], 0.0)
        self.assertEqual(s["success_rate"], 1.0)

    def test_determinism(self) -> None:
        if not self.eps:
            self.skipTest("frozen dataset not available")
        a = E.summarize(E.run_arm(self.eps, E.make_random_policy(7),
                                  E.Authority.A1_SELECT_FROM_SAFE))
        b = E.summarize(E.run_arm(self.eps, E.make_random_policy(7),
                                  E.Authority.A1_SELECT_FROM_SAFE))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
