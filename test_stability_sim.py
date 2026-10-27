"""Smoke tests for the §3.11 A1 runtime-stability sweep (stability_sim.py).

The paper sweep is 20 seeds × 1200 episodes. These tests cover the claim
reduction on hand-built rows, plus one tiny real rollout so CI never runs
that full sweep. Run: python -m unittest test_stability_sim
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import stability_sim as S


def _row(seed: int, a1_coll: float, ungated_coll: float) -> dict:
    def arm(coll: float) -> dict:
        return {
            "collision_rate": coll,
            "success_rate": 0.4,
            "spl": 0.2,
            "veto_rate": 0.1 if coll == 0.0 else 0.0,
        }
    return {"seed": seed, "a1": arm(a1_coll), "ungated": arm(ungated_coll)}


class SummarizeSeedsTests(unittest.TestCase):
    def test_claim_holds_only_when_every_a1_seed_is_zero(self) -> None:
        held = S.summarize_seeds([_row(0, 0.0, 0.8), _row(1, 0.0, 0.9)])
        self.assertTrue(held["a1_collision_always_zero"])
        self.assertEqual(held["a1"]["collision_rate"]["min"], 0.0)
        self.assertEqual(held["a1"]["collision_rate"]["max"], 0.0)
        self.assertEqual(held["a1"]["collision_rate"]["std"], 0.0)
        self.assertGreater(held["ungated"]["collision_rate"]["mean"], 0.0)

        broken = S.summarize_seeds([_row(0, 0.0, 0.8), _row(1, 0.25, 0.9)])
        self.assertFalse(broken["a1_collision_always_zero"])
        self.assertEqual(broken["a1"]["collision_rate"]["max"], 0.25)

    def test_single_seed_std_is_zero(self) -> None:
        summary = S.summarize_seeds([_row(0, 0.0, 1.0)])
        self.assertEqual(summary["ungated"]["collision_rate"]["std"], 0.0)
        self.assertEqual(summary["ungated"]["collision_rate"]["mean"], 1.0)

    def test_empty_sweep_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            S.summarize_seeds([])


class CliTests(unittest.TestCase):
    def test_defaults_match_paper_sweep(self) -> None:
        args = S.parse_args([])
        self.assertEqual(args.seeds, 20)
        self.assertEqual(args.episodes, 1200)
        self.assertEqual(args.json, "results/embodied_stability.json")
        self.assertEqual(args.figure, "figures/fig_stability.png")

    def test_reviewer_flags(self) -> None:
        args = S.parse_args(["--seeds", "3", "--episodes", "50"])
        self.assertEqual(args.seeds, 3)
        self.assertEqual(args.episodes, 50)

    def test_nonpositive_sizes_exit(self) -> None:
        with self.assertRaises(SystemExit):
            S.main(["--seeds", "0", "--episodes", "5"])
        with self.assertRaises(SystemExit):
            S.main(["--seeds", "2", "--episodes", "0"])


class TinySweepTests(unittest.TestCase):
    """A few seeds of a short prefix. Deterministic; not the 20×1200 sweep."""

    def test_a1_never_collides_and_artifacts_are_written(self) -> None:
        try:
            import embodied_sim as E
            E.episodes_from_dataset(limit=1)
        except (FileNotFoundError, ValueError):
            self.skipTest("frozen dataset not available")

        with tempfile.TemporaryDirectory() as d:
            js = os.path.join(d, "embodied_stability.json")
            fig = os.path.join(d, "fig_stability.png")
            rc = S.main(["--seeds", "2", "--episodes", "20",
                         "--json", js, "--figure", fig])
            self.assertEqual(rc, 0)
            with open(js, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["seeds"], 2)
            self.assertEqual(payload["episodes"], 20)
            self.assertEqual(payload["policy"], "random")
            self.assertTrue(payload["a1_collision_always_zero"])
            for row in payload["per_seed"]:
                self.assertEqual(row["a1"]["collision_rate"], 0.0)
                self.assertEqual(row["a1"]["terminations"]["collision"], 0)
            self.assertGreater(
                payload["summary"]["ungated"]["collision_rate"]["max"], 0.0)
            self.assertGreater(os.path.getsize(fig), 0)

            again = os.path.join(d, "again.json")
            again_fig = os.path.join(d, "again.png")
            self.assertEqual(S.main(["--seeds", "2", "--episodes", "20",
                                     "--json", again, "--figure", again_fig]), 0)
            with open(again, encoding="utf-8") as f:
                second = json.load(f)
            self.assertEqual(payload["per_seed"], second["per_seed"])


if __name__ == "__main__":
    unittest.main()
