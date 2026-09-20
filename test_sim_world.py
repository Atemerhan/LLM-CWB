"""Unit tests for the pluggable World layer (sim_world.py).

Verifies: GridWorld satisfies the World protocol and reproduces embodied_sim
(parity), the safety-by-construction property holds through the abstraction,
the kinematic-feasibility check works, and the AirSimWorld template's pure
(oracle-side) methods work in CI without a local AirSim client.

Run: python -m unittest test_sim_world
"""
from __future__ import annotations

import unittest

import embodied_sim as E
import run
import sim_world as W


class ProtocolAndParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.worlds = W.grid_worlds_from_dataset(limit=150)
            cls.eps = E.episodes_from_dataset(limit=150)
        except (FileNotFoundError, ValueError):
            cls.worlds = cls.eps = None

    def test_gridworld_is_a_world(self) -> None:
        g = [[0, 0], [0, 0]]
        self.assertIsInstance(W.GridWorld(g, (0, 0), (1, 1)), W.World)
        self.assertIsInstance(W.AirSimWorld(g, (0, 0), (1, 1)), W.World)

    def _parity(self, policy_fn, authority) -> None:
        if not self.worlds:
            self.skipTest("frozen dataset not available")
        a = E.summarize(W.run_arm(self.worlds, W.grid_policy(policy_fn), authority))
        b = E.summarize(E.run_arm(self.eps, policy_fn, authority))
        self.assertEqual(a, b)

    def test_parity_a1_greedy(self) -> None:
        self._parity(E.greedy_manhattan_policy, E.Authority.A1_SELECT_FROM_SAFE)

    def test_parity_ungated_greedy(self) -> None:
        self._parity(E.greedy_manhattan_policy, E.Authority.UNGATED)

    def test_parity_planner(self) -> None:
        self._parity(E.oracle_policy, E.Authority.PLANNER_ONLY)


class SafetyThroughAbstractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.worlds = W.grid_worlds_from_dataset(limit=150)
        except (FileNotFoundError, ValueError):
            cls.worlds = None

    def test_a1_zero_collisions(self) -> None:
        if not self.worlds:
            self.skipTest("frozen dataset not available")
        s = E.summarize(W.run_arm(self.worlds, W.grid_policy(E.make_random_policy(0)),
                                  E.Authority.A1_SELECT_FROM_SAFE))
        self.assertEqual(s["collision_rate"], 0.0)

    def test_ungated_random_can_collide(self) -> None:
        if not self.worlds:
            self.skipTest("frozen dataset not available")
        s = E.summarize(W.run_arm(self.worlds, W.grid_policy(E.make_random_policy(0)),
                                  E.Authority.UNGATED))
        self.assertGreater(s["collision_rate"], 0.0)


class KinematicLimitsTests(unittest.TestCase):
    def test_default_permissive(self) -> None:
        lim = W.KinematicLimits()
        self.assertTrue(lim.feasible(None, "up"))
        self.assertTrue(lim.feasible("right", "left"))   # reverse allowed by default

    def test_no_reverse(self) -> None:
        lim = W.KinematicLimits(allow_reverse=False)
        self.assertFalse(lim.feasible("right", "left"))  # 180° flip forbidden
        self.assertTrue(lim.feasible("right", "up"))     # 90° turn ok
        self.assertTrue(lim.feasible("right", "right"))

    def test_straight_only(self) -> None:
        lim = W.KinematicLimits(max_turn_deg=0.0)
        self.assertFalse(lim.feasible("right", "up"))
        self.assertTrue(lim.feasible("right", "right"))

    def test_world_excludes_infeasible_reverse(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]          # all open
        world = W.GridWorld(grid, (1, 1), (2, 2),
                            limits=W.KinematicLimits(allow_reverse=False))
        state = ((1, 1), "right")                          # heading east
        self.assertFalse(world.is_safe_command(state, "left"))   # reverse vetoed
        self.assertNotIn("left", world.safe_commands(state))
        self.assertTrue(world.is_safe_command(state, "down"))    # 90° ok


class AirSimTemplateTests(unittest.TestCase):
    """The oracle-side (pure) methods must work in CI without a local client."""

    def setUp(self) -> None:
        self.grid = [[0, 0], [1, 0]]
        self.world = W.AirSimWorld(self.grid, (0, 0), (0, 1))

    def test_pure_methods_without_client(self) -> None:
        state = self.world.reset()
        self.assertFalse(self.world.at_goal(state))
        self.assertTrue(self.world.is_safe_command(state, "right"))
        self.assertFalse(self.world.is_safe_command(state, "down"))   # into wall
        self.assertEqual(self.world.safe_commands(state), ["right"])
        self.assertIsNotNone(self.world.fallback_command(state))

    def test_fpv_requires_client(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.world.fpv_image()


if __name__ == "__main__":
    unittest.main()
