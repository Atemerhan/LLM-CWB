"""Smoke test for the multi-seed runtime-stability sweep (stability_sim.py).

One seed and a few frozen episodes. Calls run_sweep directly, so it does not
write results/ or figures/. Run: python -m unittest test_stability_sim
"""
from __future__ import annotations

import unittest

import stability_sim as S


class StabilitySmokeTests(unittest.TestCase):
    def test_one_seed_few_episodes(self) -> None:
        try:
            payload = S.run_sweep(1, 20)
        except (FileNotFoundError, ValueError):
            self.skipTest("frozen dataset not available")
        self.assertEqual(payload["n_seeds"], 1)
        self.assertEqual(payload["seeds"], [0])
        self.assertEqual(payload["episodes_per_seed"], 20)
        self.assertTrue(payload["deterministic_reproducible"])
        a1 = payload["aggregates"]["a1_random"]["collision_rate"]
        self.assertEqual(a1["min"], 0.0)
        self.assertEqual(a1["max"], 0.0)
        self.assertEqual(payload["per_seed"]["a1_random"][0]["collision_rate"], 0.0)
        self.assertGreater(payload["aggregates"]["ungated_random"]["collision_rate"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
