"""Unit tests for agents and the prompt builder (agents.py).

The DeepSeek HTTP path is exercised with a stubbed urlopen so no network or API
key is required. Run with: python -m unittest test_agents
"""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import agents
import run
from scoring import parse_response


class PromptTests(unittest.TestCase):
    def test_prompt_contains_grid_and_schema(self) -> None:
        sample = run.make_sample(8, 8, rng=run.random.Random(0))
        prompt = agents.build_prompt(sample)
        self.assertIn(run.render(sample), prompt)
        self.assertIn('"move"', prompt)
        self.assertIn('"confidence"', prompt)
        for d in run.DIRECTIONS:
            self.assertIn(d, prompt)

    def test_prompt_version_constant(self) -> None:
        self.assertEqual(agents.PROMPT_VERSION, "v0")


class GreedyAgentTests(unittest.TestCase):
    def test_responds_to_sample_in_protocol(self) -> None:
        sample = run.make_sample(8, 8, rng=run.random.Random(1))
        raw = agents.GreedyAgent().respond_to_sample(sample)
        parsed = parse_response(raw)
        self.assertFalse(parsed.malformed)
        self.assertEqual(parsed.confidence, 100.0)

    def test_respond_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            agents.GreedyAgent().respond("prompt")


class DeepSeekAgentTests(unittest.TestCase):
    def test_missing_key_raises(self) -> None:
        agent = agents.DeepSeekAgent(api_key=None)
        agent.api_key = None  # ensure no env leakage
        with self.assertRaises(RuntimeError):
            agent.respond("hello")

    def test_model_from_param(self) -> None:
        agent = agents.DeepSeekAgent(model="custom-model", api_key="k")
        self.assertEqual(agent.model, "custom-model")

    def test_respond_parses_openai_shape(self) -> None:
        agent = agents.DeepSeekAgent(model="m", api_key="secret")

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = {
                "choices": [
                    {"message": {"content": '{"move": "up", "confidence": 77}'}}
                ]
            }
            return _fake_response(payload)

        with mock.patch.object(agents.urllib.request, "urlopen", fake_urlopen):
            out = agent.respond("the prompt")

        self.assertEqual(out, '{"move": "up", "confidence": 77}')
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertEqual(captured["body"]["model"], "m")
        self.assertEqual(captured["body"]["messages"][0]["content"], "the prompt")
        self.assertFalse(captured["body"]["stream"])


class GLMAgentTests(unittest.TestCase):
    def test_missing_key_raises(self) -> None:
        agent = agents.GLMAgent(api_key=None)
        agent.api_key = None  # ensure no env leakage
        with self.assertRaises(RuntimeError):
            agent.respond("hello")

    def test_model_and_key_are_stripped(self) -> None:
        # A trailing newline on the key makes the Bearer header invalid; the
        # adapter must strip surrounding whitespace on both model and key.
        agent = agents.GLMAgent(model="  glm-x \n", api_key="secret\n")
        self.assertEqual(agent.model, "glm-x")
        self.assertEqual(agent.api_key, "secret")

    def test_default_base_url(self) -> None:
        agent = agents.GLMAgent(api_key="k")
        self.assertEqual(agent.base_url, agents.GLM_BASE_URL)

    def test_respond_parses_openai_shape(self) -> None:
        agent = agents.GLMAgent(model="m", api_key="secret")

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = {
                "choices": [
                    {"message": {"content": '{"move": "down", "confidence": 55}'}}
                ]
            }
            return _fake_response(payload)

        with mock.patch.object(agents.urllib.request, "urlopen", fake_urlopen):
            out = agent.respond("the prompt")

        self.assertEqual(out, '{"move": "down", "confidence": 55}')
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertEqual(captured["body"]["model"], "m")
        self.assertEqual(captured["body"]["messages"][0]["content"], "the prompt")
        self.assertFalse(captured["body"]["stream"])


class KimiAgentTests(unittest.TestCase):
    def test_missing_key_raises(self) -> None:
        agent = agents.KimiAgent(api_key=None)
        agent.api_key = None  # ensure no env leakage
        with self.assertRaises(RuntimeError):
            agent.respond("hello")

    def test_uses_kimi_defaults_not_glm(self) -> None:
        agent = agents.KimiAgent(api_key="k")
        self.assertEqual(agent.provider, "kimi")
        self.assertEqual(agent.base_url, agents.KIMI_BASE_URL)
        self.assertEqual(agent.model, agents.DEFAULT_KIMI_MODEL)

    def test_model_and_key_are_stripped(self) -> None:
        agent = agents.KimiAgent(model="  moonshot-x \n", api_key="secret\n")
        self.assertEqual(agent.model, "moonshot-x")
        self.assertEqual(agent.api_key, "secret")

    def test_respond_parses_openai_shape(self) -> None:
        agent = agents.KimiAgent(model="m", api_key="secret")

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = {
                "choices": [
                    {"message": {"content": '{"move": "left", "confidence": 42}'}}
                ]
            }
            return _fake_response(payload)

        with mock.patch.object(agents.urllib.request, "urlopen", fake_urlopen):
            out = agent.respond("the prompt")

        self.assertEqual(out, '{"move": "left", "confidence": 42}')
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertEqual(captured["body"]["model"], "m")


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_response(payload: dict) -> _FakeResp:
    return _FakeResp(json.dumps(payload).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
