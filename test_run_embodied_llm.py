"""Unit tests for the closed-loop LLM arm runner (run_embodied_llm.py).

Uses stub agents (no network). Covers sample_at, LLMPolicy parsing/caching/
malformed handling, and the A1 zero-collision invariant with a live-style policy.

Run: python -m unittest test_run_embodied_llm
"""
from __future__ import annotations

import unittest

import embodied_sim as E
import run
import sim_world as W
from agents import build_prompt
import run_embodied_llm as R


class _StubAgent:
    name = model = provider = "stub"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.seen = 0

    def respond(self, prompt: str) -> str:
        self.seen += 1
        return self._reply


class SampleAtTests(unittest.TestCase):
    def test_builds_renderable_sample(self) -> None:
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        s = R.sample_at(grid, (0, 0), (2, 2))
        self.assertEqual(s.position, (0, 0))
        self.assertEqual(s.goal, (2, 2))
        self.assertIn(s.difficulty, run.DIFFICULTIES)
        self.assertTrue(all(m in run.DIRECTIONS for m in s.correct_moves))
        self.assertIn("Grid:", build_prompt(s))


class LLMPolicyTests(unittest.TestCase):
    def test_parses_move(self) -> None:
        agent = _StubAgent('{"move":"down","confidence":80}')
        pol = R.LLMPolicy(agent)
        grid = [[0, 0], [0, 0]]
        self.assertEqual(pol(grid, (0, 0), (1, 1)), "down")
        self.assertEqual(pol.calls, 1)
        self.assertEqual(pol.malformed, 0)

    def test_caches_by_state(self) -> None:
        agent = _StubAgent('{"move":"down","confidence":80}')
        pol = R.LLMPolicy(agent)
        grid = [[0, 0], [0, 0]]
        for _ in range(5):
            pol(grid, (0, 0), (1, 1))         # identical state
        self.assertEqual(agent.seen, 1)        # queried once
        self.assertEqual(pol.calls, 1)

    def test_malformed_is_none(self) -> None:
        pol = R.LLMPolicy(_StubAgent("no json here"))
        self.assertIsNone(pol(([[0, 0], [0, 0]]), (0, 0), (1, 1)))
        self.assertEqual(pol.malformed, 1)

    def test_failed_call_is_non_aborting(self) -> None:
        class Boom:
            name = model = provider = "boom"
            def respond(self, prompt):           # noqa: ANN001
                raise RuntimeError("network down")
        pol = R.LLMPolicy(Boom())
        self.assertIsNone(pol([[0, 0], [0, 0]], (0, 0), (1, 1)))
        self.assertEqual(pol.malformed, 1)       # recorded, not raised


class InvariantTests(unittest.TestCase):
    def test_a1_zero_collision_with_live_style_policy(self) -> None:
        try:
            worlds = W.grid_worlds_from_dataset(limit=40)
        except (FileNotFoundError, ValueError):
            self.skipTest("frozen dataset not available")
        pol = W.grid_policy(R.LLMPolicy(_StubAgent('{"move":"down","confidence":99}')))
        s = E.summarize(W.run_arm(worlds, pol, E.Authority.A1_SELECT_FROM_SAFE))
        self.assertEqual(s["collision_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
