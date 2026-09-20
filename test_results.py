"""Unit tests for result storage and the results schema (results.py).

Run with: python -m unittest test_results
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import results
import run


def _dataset(n: int = 12, seed: int = 0):
    return run.generate_dataset(n, 8, 8, seed=seed)


def _correct_response(sample) -> str:
    return json.dumps({"move": sample.correct_moves[0], "confidence": 90})


class BuildResultsTests(unittest.TestCase):
    def test_top_level_shape(self) -> None:
        samples = _dataset()
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {"model": "x", "provider": "y"})
        self.assertEqual(doc["version"], results.RESULTS_VERSION)
        self.assertIn("meta", doc)
        self.assertIn("summary", doc)
        self.assertEqual(len(doc["results"]), len(samples))

    def test_meta_defaults_filled(self) -> None:
        samples = _dataset(3)
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {"model": "x"})
        meta = doc["meta"]
        self.assertIn("created_at", meta)
        self.assertEqual(meta["confidence_threshold"], 80.0)
        self.assertEqual(meta["calibration_buckets"], 10)
        self.assertEqual(meta["dataset_size"], 3)

    def test_all_correct_high_conf(self) -> None:
        samples = _dataset()
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {})
        overall = doc["summary"]["overall"]
        self.assertEqual(overall["accuracy"], 1.0)
        self.assertEqual(overall["malformed_rate"], 0.0)
        self.assertEqual(doc["summary"]["categories"]["right_high"], len(samples))

    def test_confidently_wrong_recorded(self) -> None:
        samples = _dataset(6)
        wrongs = {"up": "down", "down": "up", "left": "right", "right": "left"}
        raws = [
            json.dumps({"move": wrongs[s.correct_moves[0]], "confidence": 99})
            for s in samples
        ]
        doc = results.build_results(samples, raws, {})
        # Every answer is the reverse of a correct move; some reverses may still
        # be correct on open grids, but at least the schema records the field.
        rec = doc["results"][0]
        self.assertIn(rec["category"], analysis_categories())
        self.assertIn("confidently_wrong", doc["summary"]["overall"])

    def test_malformed_recorded(self) -> None:
        samples = _dataset(3)
        raws = ["I am not JSON" for _ in samples]
        doc = results.build_results(samples, raws, {})
        self.assertEqual(doc["summary"]["categories"]["malformed"], 3)
        self.assertEqual(doc["summary"]["overall"]["accuracy"], 0.0)
        for rec in doc["results"]:
            self.assertTrue(rec["malformed"])
            self.assertIsNone(rec["move"])
            self.assertEqual(rec["category"], "malformed")

    def test_record_fields_present(self) -> None:
        samples = _dataset(2)
        raws = [_correct_response(s) for s in samples]
        rec = results.build_results(samples, raws, {})["results"][0]
        for key in ("id", "difficulty", "position", "goal", "correct_moves",
                    "raw_response", "move", "confidence", "malformed",
                    "correct", "confident", "category"):
            self.assertIn(key, rec)

    def test_calibration_curve_present(self) -> None:
        samples = _dataset()
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {}, n_buckets=5)
        curve = doc["summary"]["calibration_curve"]
        self.assertEqual(len(curve), 5)
        self.assertIn("mean_confidence", curve[0])

    def test_length_mismatch_raises(self) -> None:
        samples = _dataset(3)
        with self.assertRaises(ValueError):
            results.build_results(samples, ["only one"], {})


class PersistenceTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        samples = _dataset(5)
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {"model": "m"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            results.save_results(path, doc)
            loaded = results.load_results(path)
        self.assertEqual(loaded["summary"]["overall"]["accuracy"],
                         doc["summary"]["overall"]["accuracy"])

    def test_bad_version_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"version": 999}')
            with self.assertRaises(ValueError):
                results.load_results(path)


class ReportTests(unittest.TestCase):
    def test_report_is_string(self) -> None:
        samples = _dataset(6)
        raws = [_correct_response(s) for s in samples]
        doc = results.build_results(samples, raws, {"model": "m", "provider": "p"})
        report = results.format_report(doc)
        self.assertIn("accuracy", report)
        self.assertIn("calibration", report)


def analysis_categories():
    import analysis
    return set(analysis.CATEGORIES) | {"malformed"}


if __name__ == "__main__":
    unittest.main()
