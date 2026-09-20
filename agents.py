"""Agents that answer next-move queries, plus the v0 prompt builder.

An Agent maps a prompt string to a raw response string. Two backends:

- `GreedyAgent`: the wall-blind baseline, wrapped as an Agent so the whole
  measurement pipeline can be exercised offline (no network, no key).
- `DeepSeekAgent`: queries DeepSeek's OpenAI-compatible chat API.
- `GLMAgent`: queries Zhipu GLM's OpenAI-compatible chat API (same request /
  response shape as DeepSeek; only the queried model differs).
- `KimiAgent`: queries Moonshot Kimi's OpenAI-compatible chat API (a thin
  subclass of GLMAgent with its own KIMI_* credentials/endpoint).

The prompt here is deliberately minimal ("v0") — prompt optimization is a
later step. The goal now is a correct, end-to-end measurement path.

Python 3.11 compatible.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from baseline import greedy_response
from run import Sample, render

PROMPT_VERSION = "v0"

# DeepSeek exposes an OpenAI-compatible endpoint.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# NOTE: set the exact model id your account exposes via DEEPSEEK_MODEL.
# "deepseek-chat" is the default OpenAI-compatible alias; override as needed.
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

# GLM (Zhipu AI) exposes an OpenAI-compatible v4 endpoint. Two regional
# platforms exist with separate keys; the default below is the Zhipu BigModel
# (China) host. For the international Z.ai platform set GLM_BASE_URL to
# "https://api.z.ai/api/paas/v4".
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
# NOTE: set the exact model id your account exposes via GLM_MODEL.
DEFAULT_GLM_MODEL = "glm-4-plus"
# Transient infrastructure failures worth retrying (vs. 4xx client errors,
# which are raised immediately). 429 (rate-limit) is included so a provider
# RPM cap is ridden out via backoff rather than dropping the sample. Connection
# resets and timeouts are also retried; see GLMAgent.respond.
GLM_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Kimi (Moonshot AI) exposes an OpenAI-compatible endpoint. Default is the
# Moonshot China host; override via KIMI_BASE_URL for the international host.
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
# NOTE: set the exact model id your account exposes via KIMI_MODEL.
DEFAULT_KIMI_MODEL = "moonshot-v1-8k"
# Moonshot trial orgs cap at ~20 requests/min. Throttle to stay under the cap
# (1 request / KIMI_MIN_INTERVAL s) so the run is not decimated by 429s; the
# pacing changes only timing, not the prompt/response/scoring. Override via
# the KIMI_MIN_INTERVAL env var.
DEFAULT_KIMI_MIN_INTERVAL = 3.5


def build_prompt(sample: Sample) -> str:
    """Construct the v0 (un-optimized) prompt for one sample."""
    grid = render(sample)
    return (
        "You are an agent navigating a 2D grid maze.\n"
        "Legend: '.' = open cell, '#' = wall, 'A' = your current position, "
        "'G' = the goal.\n"
        "You may move exactly one cell: up, down, left, or right "
        "(no diagonals). You cannot move into a wall or off the grid.\n"
        "Choose the single move that gets you closer to the goal along the "
        "shortest path.\n\n"
        f"Grid:\n{grid}\n\n"
        'Respond with ONLY a JSON object of the form '
        '{"move": "<up|down|left|right>", "confidence": <integer 0-100>}.\n'
        '"confidence" is how certain you are the move is correct.'
    )


class Agent(Protocol):
    """Anything that turns a prompt into a raw response string."""

    name: str

    def respond(self, prompt: str) -> str:
        ...


class GreedyAgent:
    """Wall-blind greedy baseline as an Agent (offline pipeline check).

    It ignores the prompt text and answers from the sample directly, so it must
    be driven via `respond_to_sample`. `respond` exists only to satisfy the
    protocol and raises if used.
    """

    name = "greedy"
    provider = "greedy"
    model = "greedy-baseline"

    def respond_to_sample(self, sample: Sample) -> str:
        return greedy_response(sample)

    def respond(self, prompt: str) -> str:  # pragma: no cover - not used
        raise NotImplementedError("GreedyAgent answers from samples, not prompts")


class DeepSeekAgent:
    """Queries DeepSeek's OpenAI-compatible /chat/completions endpoint.

    Reads the API key from `api_key` or the DEEPSEEK_API_KEY env var, and the
    model id from `model` or the DEEPSEEK_MODEL env var. Uses stdlib urllib so
    no extra dependency is required.
    """

    name = "deepseek"
    provider = "deepseek"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def respond(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set; cannot call the DeepSeek API"
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {detail}") from exc
        return body["choices"][0]["message"]["content"]


class GLMAgent:
    """Queries Zhipu GLM's OpenAI-compatible /chat/completions endpoint.

    Reads the API key from `api_key` or the GLM_API_KEY env var, the model id
    from `model` or the GLM_MODEL env var (default ``glm-4-plus``), and the
    base URL from `base_url` or the GLM_BASE_URL env var (default the Zhipu
    BigModel v4 host). The request/response shape is identical to the DeepSeek
    adapter (both OpenAI-compatible), so the v0 prompt, response parsing,
    scoring, and confidence extraction are reused without any change — only the
    model being queried differs. Uses stdlib urllib (no extra dependency).
    """

    name = "glm"
    provider = "glm"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        min_interval: float = 0.0,
    ) -> None:
        self.model = (model or os.environ.get("GLM_MODEL", DEFAULT_GLM_MODEL)).strip()
        api_key = api_key or os.environ.get("GLM_API_KEY")
        # Tolerate keys pasted with surrounding whitespace/newlines: a trailing
        # newline makes the "Bearer <key>" Authorization header invalid.
        self.api_key = api_key.strip() if api_key else api_key
        self.base_url = (
            base_url or os.environ.get("GLM_BASE_URL", GLM_BASE_URL)
        ).strip().rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Minimum seconds between successive requests (client-side rate limit).
        # 0 = no throttle (GLM/DeepSeek default); >0 paces requests (Kimi).
        self.min_interval = min_interval
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.min_interval > 0:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def respond(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GLM_API_KEY is not set; cannot call the GLM API")
        self._throttle()
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": False,
            }
        ).encode("utf-8")

        # Retry ONLY transient infrastructure failures (HTTP 5xx, connection
        # reset, timeout) with exponential backoff, up to `max_retries` times.
        # Client errors (4xx) and any other failure are raised immediately.
        # A retry re-sends the identical request, so prompts, scoring, and
        # confidence handling are unaffected.
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:  # pragma: no cover - network path
                if exc.code in GLM_RETRY_STATUS and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                detail = exc.read().decode("utf-8", "replace")
                raise RuntimeError(f"GLM API error {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:  # pragma: no cover - network path
                # HTTPError is handled above; here `reason` is the underlying
                # OSError. Retry only connection reset / timeout.
                if (
                    isinstance(exc.reason, (TimeoutError, ConnectionResetError))
                    and attempt < self.max_retries
                ):
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                raise RuntimeError(f"GLM API request failed: {exc.reason}") from exc
            except (TimeoutError, ConnectionResetError) as exc:  # pragma: no cover
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                raise RuntimeError(f"GLM API request failed: {exc}") from exc


class KimiAgent(GLMAgent):
    """Queries Moonshot Kimi's OpenAI-compatible /chat/completions endpoint.

    Behaviourally identical to GLMAgent (same request/response shape, same
    transient-failure retry and credential stripping); only the endpoint,
    credentials, and model differ. Reuses GLMAgent.respond unchanged and
    resolves its own KIMI_* env vars with no GLM fallback, so the frozen v0
    prompt, parsing, scoring, and confidence handling are reused as-is.
    """

    name = "kimi"
    provider = "kimi"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        min_interval: float | None = None,
    ) -> None:
        self.model = (model or os.environ.get("KIMI_MODEL", DEFAULT_KIMI_MODEL)).strip()
        api_key = api_key or os.environ.get("KIMI_API_KEY")
        self.api_key = api_key.strip() if api_key else api_key
        self.base_url = (
            base_url or os.environ.get("KIMI_BASE_URL", KIMI_BASE_URL)
        ).strip().rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if min_interval is None:
            min_interval = float(
                os.environ.get("KIMI_MIN_INTERVAL", DEFAULT_KIMI_MIN_INTERVAL)
            )
        self.min_interval = min_interval
        self._last_call = 0.0

    def respond(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("KIMI_API_KEY is not set; cannot call the Kimi API")
        return super().respond(prompt)
