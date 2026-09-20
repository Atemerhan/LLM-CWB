"""Trivial baseline agent: always move toward the goal (ignoring walls).

This is the validity check demanded by the audit. If a wall-blind greedy agent
scores high overall, the dataset is too easy and a real model could look
competent without reasoning. We expect: near-perfect on `easy` (straight-line)
samples, and markedly worse on `hard` (detour-forcing) samples. That gap is
what makes the benchmark able to surface *confidently wrong* behavior.

The baseline emits the same protocol as any agent (a JSON move + confidence),
so it flows through the exact same parsing/scoring path as a future LLM.

No LLM API is used here. Python 3.11 compatible.
"""

from __future__ import annotations

import json
import sys

from run import DEFAULT_DATASET_PATH, DIFFICULTIES, Sample, load_dataset
from scoring import Score, aggregate, parse_response, score


def greedy_move(sample: Sample) -> str:
    """Pick the direction that reduces the larger coordinate gap to the goal.

    Wall-blind on purpose: it never consults the grid, only the goal vector.
    """
    pr, pc = sample.position
    gr, gc = sample.goal
    drow, dcol = gr - pr, gc - pc

    if abs(drow) >= abs(dcol):
        if drow != 0:
            return "down" if drow > 0 else "up"
        return "right" if dcol > 0 else "left"
    if dcol != 0:
        return "right" if dcol > 0 else "left"
    return "down" if drow > 0 else "up"


def greedy_response(sample: Sample) -> str:
    """Baseline response in the confidence protocol (max confidence)."""
    return json.dumps({"move": greedy_move(sample), "confidence": 100})


def benchmark(samples: list[Sample]) -> dict:
    """Run the greedy baseline over samples; aggregate overall and per bucket."""
    by_bucket: dict[str, list[Score]] = {d: [] for d in DIFFICULTIES}
    all_scores: list[Score] = []
    for sample in samples:
        parsed = parse_response(greedy_response(sample))
        result = score(sample, parsed)
        all_scores.append(result)
        by_bucket[sample.difficulty].append(result)

    return {
        "overall": aggregate(all_scores),
        "by_difficulty": {d: aggregate(s) for d, s in by_bucket.items()},
    }


def _fmt(metrics: dict) -> str:
    if metrics.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={metrics['n']:3d}  "
        f"accuracy={metrics['accuracy']:.2f}  "
        f"malformed={metrics['malformed_rate']:.2f}"
    )


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    samples = load_dataset(path)
    results = benchmark(samples)

    print(f"Trivial baseline (greedy, wall-blind) on {path}\n")
    print(f"  overall      {_fmt(results['overall'])}")
    for difficulty in DIFFICULTIES:
        print(f"  {difficulty:<12} {_fmt(results['by_difficulty'][difficulty])}")
    print(
        "\nInterpretation: high 'easy' accuracy with a clear drop on 'hard' "
        "confirms the dataset separates greedy luck from real navigation."
    )


if __name__ == "__main__":
    main()
