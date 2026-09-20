"""Result storage: the exact JSON schema for an evaluation run.

A results file captures everything needed to audit and re-analyze a run without
re-querying the model: run metadata, every raw response, the parsed/scored
outcome per sample, and aggregate + calibration summaries.

Top-level schema (RESULTS_VERSION = 1):

{
  "version": 1,
  "meta": {
    "model": "deepseek-chat",          # model identifier actually queried
    "provider": "deepseek" | "greedy", # agent backend
    "created_at": "2026-06-05T12:00:00Z",
    "dataset_path": "data/eval_dataset.json",
    "dataset_version": 1,
    "dataset_size": 1200,
    "prompt_version": "v0",
    "temperature": 0.0,
    "confidence_threshold": 80.0,
    "calibration_buckets": 10
  },
  "summary": {
    "overall":   { ... metrics ... , "ece": 0.0, "categories": {...} },
    "by_difficulty": { "easy": {...}, "medium": {...}, "hard": {...} },
    "categories": { "right_high":N, "right_low":N, "wrong_low":N,
                    "wrong_high":N, "malformed":N },
    "calibration_curve": [ { "bucket":"[0,10)", "low":0, "high":10,
                             "n":N, "accuracy":x, "mean_confidence":y }, ... ]
  },
  "results": [
    {
      "id": 0,
      "difficulty": "easy",
      "position": [r, c],
      "goal": [r, c],
      "correct_moves": ["down"],
      "raw_response": "...verbatim model text...",
      "move": "down" | null,
      "confidence": 90.0 | null,
      "malformed": false,
      "correct": true,
      "confident": true,
      "category": "right_high"
    },
    ...
  ]
}

No LLM API is used here. Python 3.11 compatible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from analysis import (
    bucket_to_dict,
    calibration_curve,
    categorize,
    category_counts,
    expected_calibration_error,
)
from run import DIFFICULTIES, Sample
from scoring import CONFIDENCE_THRESHOLD, Score, aggregate, parse_response, score

RESULTS_VERSION = 1


def _summarize(
    scores: list[Score], threshold: float, n_buckets: int
) -> dict:
    summary = aggregate(scores)
    summary["ece"] = expected_calibration_error(scores, n_buckets)
    summary["categories"] = category_counts(scores, threshold)
    return summary


def build_results(
    samples: list[Sample],
    raw_responses: list[str],
    meta: dict,
    threshold: float = CONFIDENCE_THRESHOLD,
    n_buckets: int = 10,
) -> dict:
    """Parse + score every response and assemble the full results document.

    `raw_responses[i]` is the verbatim text from the agent for `samples[i]`.
    """
    if len(samples) != len(raw_responses):
        raise ValueError("samples and raw_responses must be the same length")

    records: list[dict] = []
    scores: list[Score] = []
    by_difficulty: dict[str, list[Score]] = {d: [] for d in DIFFICULTIES}

    for i, (sample, raw) in enumerate(zip(samples, raw_responses)):
        parsed = parse_response(raw)
        result = score(sample, parsed, threshold)
        scores.append(result)
        by_difficulty[sample.difficulty].append(result)
        records.append(
            {
                "id": i,
                "difficulty": sample.difficulty,
                "position": list(sample.position),
                "goal": list(sample.goal),
                "correct_moves": sample.correct_moves,
                "raw_response": raw,
                "move": parsed.move,
                "confidence": parsed.confidence,
                "malformed": result.malformed,
                "correct": result.correct,
                "confident": result.confident,
                "category": categorize(result, threshold),
            }
        )

    full_meta = dict(meta)
    full_meta.setdefault(
        "created_at",
        datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    full_meta.setdefault("confidence_threshold", threshold)
    full_meta.setdefault("calibration_buckets", n_buckets)
    full_meta.setdefault("dataset_size", len(samples))

    summary = {
        "overall": _summarize(scores, threshold, n_buckets),
        "by_difficulty": {
            d: _summarize(by_difficulty[d], threshold, n_buckets)
            for d in DIFFICULTIES
        },
        "categories": category_counts(scores, threshold),
        "calibration_curve": [
            bucket_to_dict(b) for b in calibration_curve(scores, n_buckets)
        ],
    }

    return {
        "version": RESULTS_VERSION,
        "meta": full_meta,
        "summary": summary,
        "results": records,
    }


def save_results(path: str, document: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    if document.get("version") != RESULTS_VERSION:
        raise ValueError(f"unsupported results version: {document.get('version')!r}")
    return document


def format_report(document: dict) -> str:
    """Human-readable summary of a results document for the console."""
    meta = document["meta"]
    summary = document["summary"]
    overall = summary["overall"]
    cats = summary["categories"]

    lines = [
        f"model={meta.get('model')} provider={meta.get('provider')} "
        f"prompt={meta.get('prompt_version')} n={overall['n']}",
        "",
        f"  accuracy            {overall['accuracy']:.3f}",
        f"  malformed_rate      {overall['malformed_rate']:.3f}",
        f"  mean_confidence     {_fmt(overall.get('mean_confidence'))}",
        f"  ECE                 {_fmt(overall.get('ece'))}",
        f"  confidently_wrong   {overall['confidently_wrong']} "
        f"({overall['confidently_wrong_rate']:.3f} of well-formed)",
        "",
        "  categories:",
        f"    right/high  {cats['right_high']:5d}    right/low  {cats['right_low']:5d}",
        f"    wrong/high  {cats['wrong_high']:5d}    wrong/low  {cats['wrong_low']:5d}"
        f"    malformed  {cats['malformed']:5d}",
        "",
        "  accuracy by difficulty:",
    ]
    for d in DIFFICULTIES:
        s = summary["by_difficulty"][d]
        if s["n"] == 0:
            lines.append(f"    {d:<7} n=   0  (no samples)")
            continue
        lines.append(
            f"    {d:<7} n={s['n']:4d}  acc={s['accuracy']:.3f}  "
            f"ece={_fmt(s.get('ece'))}  cw={s['confidently_wrong']}"
        )

    lines.append("")
    lines.append("  calibration curve (confidence -> accuracy):")
    lines.append("    bucket      n     acc    mean_conf")
    for b in summary["calibration_curve"]:
        acc = _fmt(b["accuracy"])
        mc = _fmt(b["mean_confidence"])
        lines.append(f"    {b['bucket']:<9} {b['n']:5d}  {acc:>6}  {mc:>9}")
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"
