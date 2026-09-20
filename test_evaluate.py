"""Unit tests for the evaluation runner (evaluate.py).

Run with: python -m unittest test_evaluate
"""

from __future__ import annotations

import json
import unittest

import evaluate
import run


class FakePromptAgent:
    """Agent that echoes a fixed move with high confidence (uses prompts)."""

    name = "fake"
    provider = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def respond(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps({"move": "up", "confidence": 90})


class CollectResponsesTests(unittest.TestCase):
    def test_greedy_uses_sample_path(self) -> None:
        samples = run.generate_dataset(6, 8, 8, seed=0)
        agent = evaluate.GreedyAgent()
        raws = evaluate.collect_responses(agent, samples)
        self.assertEqual(len(raws), 6)
        for raw in raws:
            self.assertIn('"move"', raw)

    def test_prompt_agent_called_per_sample(self) -> None:
        samples = run.generate_dataset(4, 8, 8, seed=1)
        agent = FakePromptAgent()
        raws = evaluate.collect_responses(agent, samples)
        self.assertEqual(agent.calls, 4)
        self.assertEqual(len(raws), 4)

    def test_non_aborting_on_permanent_failure(self) -> None:
        # An agent whose 3rd call raises must not abort the run; the failed
        # sample is recorded as an empty (malformed) response and the rest
        # continue, preserving one response per sample.
        class FlakyAgent:
            name = provider = "flaky"
            model = "flaky-model"

            def __init__(self) -> None:
                self.calls = 0

            def respond(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("API request failed: read timed out")
                return json.dumps({"move": "up", "confidence": 90})

        samples = run.generate_dataset(5, 8, 8, seed=2)
        raws = evaluate.collect_responses(FlakyAgent(), samples)
        self.assertEqual(len(raws), 5)
        self.assertEqual(raws[2], "")  # failed sample -> empty -> malformed


class MakeAgentTests(unittest.TestCase):
    def test_greedy(self) -> None:
        self.assertIsInstance(evaluate.make_agent("greedy", 0.0), evaluate.GreedyAgent)

    def test_deepseek(self) -> None:
        self.assertIsInstance(
            evaluate.make_agent("deepseek", 0.0), evaluate.DeepSeekAgent
        )

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate.make_agent("nope", 0.0)


class EndToEndTests(unittest.TestCase):
    def test_greedy_pipeline_builds_results(self) -> None:
        samples = run.generate_dataset(30, 8, 8, seed=2)
        agent = evaluate.GreedyAgent()
        raws = evaluate.collect_responses(agent, samples)
        from results import build_results

        doc = build_results(samples, raws, {"model": "greedy", "provider": "greedy"})
        self.assertEqual(doc["summary"]["overall"]["n"], 30)
        self.assertEqual(doc["summary"]["overall"]["malformed_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
