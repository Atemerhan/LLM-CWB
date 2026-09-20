"""End-to-end evaluation runner: dataset -> prompt -> agent -> score -> results.

Loads the committed dataset, queries an agent for each sample, then parses,
scores, and writes a results JSON document (see results.py for the schema) plus
a console calibration report.

Usage:
  # Offline pipeline check with the wall-blind greedy baseline:
  python evaluate.py --agent greedy --output results/greedy.json

  # Live DeepSeek run (requires DEEPSEEK_API_KEY, and DEEPSEEK_MODEL if your
  # account's model id differs from the default):
  python evaluate.py --agent deepseek --output results/deepseek.json

  # Smoke test on a handful of samples:
  python evaluate.py --agent greedy --limit 5

No prompt optimization is attempted here — this is the measurement harness.
Python 3.11 compatible.
"""

from __future__ import annotations

import argparse
import os

from agents import (
    PROMPT_VERSION,
    DeepSeekAgent,
    GLMAgent,
    GreedyAgent,
    KimiAgent,
    build_prompt,
)
from results import build_results, format_report, save_results
from run import DEFAULT_DATASET_PATH, load_dataset
from scoring import CONFIDENCE_THRESHOLD


def collect_responses(agent, samples) -> list[str]:
    """Query the agent for every sample, returning verbatim responses.

    GreedyAgent answers from the sample; other agents receive the built prompt.
    Non-aborting: a permanent per-sample failure (e.g. an API timeout that
    survives the adapter's retries) is recorded as an empty response — which the
    frozen parser scores as malformed (a no-decision) — so one bad call cannot
    discard a long run. The prompt, parser, scoring, and confidence handling are
    unchanged; only run robustness improves.
    """
    raws: list[str] = []
    use_sample = hasattr(agent, "respond_to_sample")
    total = len(samples)
    failures = 0
    for i, sample in enumerate(samples, 1):
        try:
            if use_sample:
                raws.append(agent.respond_to_sample(sample))
            else:
                raws.append(agent.respond(build_prompt(sample)))
        except Exception as exc:  # non-aborting: record no-decision, continue
            failures += 1
            raws.append("")  # empty -> parse_response marks malformed
            print(f"\n  [warn] sample {i}/{total} failed permanently: {exc}",
                  flush=True)
        if i % 50 == 0 or i == total:
            print(f"  queried {i}/{total}", end="\r", flush=True)
    print()
    if failures:
        print(f"  permanent failures recorded as malformed: {failures}/{total}")
    return raws


def make_agent(name: str, temperature: float):
    if name == "greedy":
        return GreedyAgent()
    if name == "deepseek":
        return DeepSeekAgent(temperature=temperature)
    if name == "glm":
        return GLMAgent(temperature=temperature)
    if name == "kimi":
        return KimiAgent(temperature=temperature)
    raise ValueError(f"unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the next-move benchmark.")
    parser.add_argument("--agent", choices=["greedy", "deepseek", "glm", "kimi"], default="greedy")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", default=None, help="results JSON path")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N samples")
    parser.add_argument("--balanced", type=int, default=None,
                        help="use first N samples per difficulty (stratified subset)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    if args.balanced is not None:
        from collections import defaultdict
        by = defaultdict(list)
        for s in samples:
            by[s.difficulty].append(s)
        samples = [s for d in ("easy", "medium", "hard")
                   for s in by[d][: args.balanced]]
    if args.limit is not None:
        samples = samples[: args.limit]

    agent = make_agent(args.agent, args.temperature)
    print(f"Evaluating {len(samples)} samples with agent={args.agent} "
          f"(model={getattr(agent, 'model', agent.name)})")

    raw_responses = collect_responses(agent, samples)

    meta = {
        "model": getattr(agent, "model", agent.name),
        "provider": getattr(agent, "provider", agent.name),
        "dataset_path": args.dataset,
        "dataset_version": 1,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
    }
    document = build_results(samples, raw_responses, meta,
                             threshold=args.threshold, n_buckets=args.buckets)

    print()
    print(format_report(document))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_results(args.output, document)
        print(f"\nwrote results -> {args.output}")


if __name__ == "__main__":
    main()
