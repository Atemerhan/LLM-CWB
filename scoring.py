"""Response protocol, malformed-response policy, and scoring.

The confidence protocol (the contract every agent — baseline or LLM — must
satisfy) is a single JSON object:

    {"move": "up", "confidence": 73}

- `move` must be one of up/down/left/right (case-insensitive, trimmed).
- `confidence` must be a number in [0, 100].

`parse_response` extracts and validates this from raw text and never raises on
bad input — it returns a `ParsedResponse` with `malformed=True` instead.

Malformed-response policy (see SPEC.md):
  A malformed response is counted as an INCORRECT answer for accuracy, is
  tracked separately via a malformed rate, and is EXCLUDED from
  confidence-calibration and "confidently wrong" statistics (a response with no
  trustworthy move/confidence cannot be confidently wrong).

No LLM API is used here. Python 3.11 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from run import DIRECTIONS, Sample

# A response is "confident" at or above this stated confidence.
CONFIDENCE_THRESHOLD = 80.0

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedResponse:
    """Result of parsing a raw agent response against the protocol."""

    move: str | None
    confidence: float | None
    malformed: bool
    error: str | None
    raw: str


def _malformed(raw: str, error: str) -> ParsedResponse:
    return ParsedResponse(move=None, confidence=None, malformed=True,
                          error=error, raw=raw)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from possibly-noisy text.

    Tries a direct parse first, then the first {...} span (handles models that
    wrap JSON in prose or ```json fences). Returns None if nothing parses to a
    dict.
    """
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    match = _JSON_OBJECT_RE.search(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_response(text: str) -> ParsedResponse:
    """Parse and validate a raw response against the confidence protocol."""
    raw = text if isinstance(text, str) else str(text)
    obj = _extract_json_object(raw)
    if obj is None:
        return _malformed(raw, "no JSON object found")

    if "move" not in obj or "confidence" not in obj:
        return _malformed(raw, "missing 'move' or 'confidence' key")

    move = obj["move"]
    if not isinstance(move, str):
        return _malformed(raw, "'move' is not a string")
    move = move.strip().lower()
    if move not in DIRECTIONS:
        return _malformed(raw, f"invalid move: {obj['move']!r}")

    confidence = obj["confidence"]
    # Reject bools (bool is an int subclass) and non-numbers.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return _malformed(raw, f"invalid confidence type: {obj['confidence']!r}")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 100.0:
        return _malformed(raw, f"confidence out of range: {confidence}")

    return ParsedResponse(move=move, confidence=confidence, malformed=False,
                          error=None, raw=raw)


@dataclass(frozen=True)
class Score:
    """Per-sample scoring outcome."""

    correct: bool
    malformed: bool
    confident: bool
    confidently_wrong: bool
    confidence: float | None


def score(
    sample: Sample,
    parsed: ParsedResponse,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Score:
    """Score one parsed response against a sample's ground truth.

    - correct: move is one of the sample's correct moves (malformed => False).
    - confident: stated confidence >= threshold (malformed => False).
    - confidently_wrong: confident AND not correct AND not malformed.
    """
    if parsed.malformed:
        return Score(correct=False, malformed=True, confident=False,
                     confidently_wrong=False, confidence=None)

    correct = parsed.move in sample.correct_moves
    confident = parsed.confidence >= threshold
    return Score(
        correct=correct,
        malformed=False,
        confident=confident,
        confidently_wrong=confident and not correct,
        confidence=parsed.confidence,
    )


def aggregate(scores: list[Score]) -> dict:
    """Summarize a list of scores into headline metrics.

    Accuracy counts malformed responses as incorrect (task failures).
    Calibration metrics (mean confidence, confidently-wrong rate) are computed
    over well-formed responses only, per the malformed policy.
    """
    n = len(scores)
    if n == 0:
        return {"n": 0}

    well_formed = [s for s in scores if not s.malformed]
    correct = sum(s.correct for s in scores)
    confidently_wrong = sum(s.confidently_wrong for s in scores)
    malformed = sum(s.malformed for s in scores)
    confidences = [s.confidence for s in well_formed if s.confidence is not None]

    return {
        "n": n,
        "accuracy": correct / n,
        "malformed_rate": malformed / n,
        "well_formed": len(well_formed),
        "confidently_wrong": confidently_wrong,
        "confidently_wrong_rate": (
            confidently_wrong / len(well_formed) if well_formed else 0.0
        ),
        "mean_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
    }
