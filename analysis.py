"""Confidence calibration analysis and four-category breakdown.

Built on top of `scoring.Score`. Provides:

- The four canonical categories (over well-formed responses):
    right/high, right/low, wrong/low, wrong/high
  where "high"/"low" is split by the confidence threshold and
  wrong/high == "confidently wrong".
- A calibration curve: accuracy vs. mean confidence per confidence bucket.
- Expected Calibration Error (ECE): the bucket-weighted gap between stated
  confidence and observed accuracy.

A well-calibrated model's accuracy within a confidence bucket should roughly
equal the bucket's confidence (e.g. answers given ~90% confidence are right
~90% of the time). Confidently-wrong behavior shows up as high-confidence
buckets with low accuracy.

No LLM API is used here. Python 3.11 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring import CONFIDENCE_THRESHOLD, Score

# The four quadrants plus the out-of-band malformed category.
CATEGORIES: tuple[str, ...] = ("right_high", "right_low", "wrong_low", "wrong_high")


def categorize(score: Score, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """Map a per-sample score to one of the four categories, or 'malformed'.

    'high'/'low' is split at `threshold`; wrong_high is the confidently-wrong
    quadrant. Malformed responses have no trustworthy confidence and are their
    own category, excluded from the 2x2.
    """
    if score.malformed:
        return "malformed"
    high = score.confidence is not None and score.confidence >= threshold
    if score.correct:
        return "right_high" if high else "right_low"
    return "wrong_high" if high else "wrong_low"


def category_counts(
    scores: list[Score], threshold: float = CONFIDENCE_THRESHOLD
) -> dict[str, int]:
    """Count the four categories plus 'malformed' across scores."""
    counts = {name: 0 for name in CATEGORIES}
    counts["malformed"] = 0
    for s in scores:
        counts[categorize(s, threshold)] += 1
    return counts


@dataclass(frozen=True)
class Bucket:
    """One confidence bucket on the calibration curve."""

    label: str
    low: float
    high: float
    n: int
    accuracy: float | None        # over well-formed responses in the bucket
    mean_confidence: float | None


def _bucket_edges(n_buckets: int) -> list[tuple[float, float]]:
    width = 100.0 / n_buckets
    return [(i * width, (i + 1) * width) for i in range(n_buckets)]


def calibration_curve(
    scores: list[Score], n_buckets: int = 10
) -> list[Bucket]:
    """Bucket well-formed responses by stated confidence and report accuracy.

    Confidence in [low, high); the final bucket includes 100. Malformed
    responses (no confidence) are excluded.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")

    well_formed = [
        s for s in scores if not s.malformed and s.confidence is not None
    ]
    edges = _bucket_edges(n_buckets)
    buckets: list[Bucket] = []
    for i, (low, high) in enumerate(edges):
        is_last = i == n_buckets - 1
        members = [
            s for s in well_formed
            if low <= s.confidence < high or (is_last and s.confidence == 100.0)
        ]
        n = len(members)
        label = f"[{low:.0f},{high:.0f}{']' if is_last else ')'}"
        if n == 0:
            buckets.append(Bucket(label, low, high, 0, None, None))
        else:
            acc = sum(s.correct for s in members) / n
            mean_conf = sum(s.confidence for s in members) / n
            buckets.append(Bucket(label, low, high, n, acc, mean_conf))
    return buckets


def expected_calibration_error(
    scores: list[Score], n_buckets: int = 10
) -> float | None:
    """Bucket-weighted mean gap between confidence and accuracy (0..1).

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b/100|, over
    well-formed responses. None if there are no well-formed responses.
    """
    buckets = calibration_curve(scores, n_buckets)
    total = sum(b.n for b in buckets)
    if total == 0:
        return None
    ece = 0.0
    for b in buckets:
        if b.n == 0 or b.accuracy is None or b.mean_confidence is None:
            continue
        ece += (b.n / total) * abs(b.accuracy - b.mean_confidence / 100.0)
    return ece


def bucket_to_dict(b: Bucket) -> dict:
    return {
        "bucket": b.label,
        "low": b.low,
        "high": b.high,
        "n": b.n,
        "accuracy": b.accuracy,
        "mean_confidence": b.mean_confidence,
    }
