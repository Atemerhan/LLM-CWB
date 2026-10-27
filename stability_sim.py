#!/usr/bin/env python3
"""Multi-seed runtime stability of the A1 authority gate (paper §3.11).

The kinematic closed loop already shows a collision rate of 0 under A1 for a
single random controller (`embodied_sim`). This sweep asks whether that zero
is an accident of one seed: the same frozen-episode prefix is rolled out
under a fresh `make_random_policy(seed)` for every seed, once ungated and
once under A1.

Claim: the A1 collision rate is exactly 0 on every seed. The ungated arm is
the control that shows the harness can still observe collisions. Success and
SPL may move with the seed; the collision rate under A1 does not.

The sweep cited with the thesis is 20 seeds × 1200 frozen episodes. A
defense reviewer can check the same claim on a small prefix:

    python stability_sim.py --seeds 3 --episodes 50

Writes `results/embodied_stability.json` and `figures/fig_stability.png`.
No model calls, and the frozen baselines are not modified. Python 3.11
compatible; rendering the figure needs matplotlib (Agg backend).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Optional

import embodied_sim as E

DEFAULT_SEEDS = 20
DEFAULT_EPISODES = 1200
DEFAULT_JSON = "results/embodied_stability.json"
DEFAULT_FIGURE = "figures/fig_stability.png"

# Cross-seed summary. Collision is the safety claim; the others show that
# performance is allowed to vary while safety does not.
_SUMMARY_KEYS = ("collision_rate", "success_rate", "spl", "veto_rate")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A1 runtime-stability sweep across random-policy seeds "
                    "(paper §3.11).",
    )
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                   help=f"number of random-policy seeds (default {DEFAULT_SEEDS})")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                   help="frozen episodes per seed, taken as a prefix of the "
                        f"dataset (default {DEFAULT_EPISODES}, the full set)")
    p.add_argument("--json", default=DEFAULT_JSON,
                   help=f"machine-readable sweep (default {DEFAULT_JSON})")
    p.add_argument("--figure", default=DEFAULT_FIGURE,
                   help=f"collision-vs-seed figure (default {DEFAULT_FIGURE})")
    return p.parse_args(argv)


def _std(values: list[float]) -> float:
    """Sample standard deviation; 0 when a single seed leaves nothing to vary."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarize_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / sample-std / min / max of each arm across seeds.

    `a1_collision_always_zero` is the §3.11 claim: every seed's A1 collision
    rate is exactly 0, not merely a mean that rounds to 0.
    """
    if not per_seed:
        raise ValueError("per_seed is empty")

    def block(arm: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key in _SUMMARY_KEYS:
            vals = [float(row[arm][key]) for row in per_seed]
            out[key] = {
                "mean": sum(vals) / len(vals),
                "std": _std(vals),
                "min": min(vals),
                "max": max(vals),
            }
        return out

    a1_coll = [float(row["a1"]["collision_rate"]) for row in per_seed]
    return {
        "a1_collision_always_zero": all(c == 0.0 for c in a1_coll),
        "a1": block("a1"),
        "ungated": block("ungated"),
    }


def run_sweep(seeds: int, episodes: int) -> dict[str, Any]:
    """Roll the frozen prefix out under one random policy per seed.

    Each seed builds two independent controllers (`make_random_policy(seed)`),
    so the ungated arm and the A1 arm both start from that seed. Episode order
    is the frozen dataset prefix and does not depend on the seed — the only
    thing that changes across the sweep is the high-level controller.
    """
    if seeds < 1 or episodes < 1:
        raise ValueError("seeds and episodes must be >= 1")
    eps = E.episodes_from_dataset(limit=episodes)
    if not eps:
        raise ValueError("frozen dataset produced no episodes")

    per_seed: list[dict[str, Any]] = []
    for seed in range(seeds):
        ungated = E.summarize(E.run_arm(
            eps, E.make_random_policy(seed), E.Authority.UNGATED))
        a1 = E.summarize(E.run_arm(
            eps, E.make_random_policy(seed), E.Authority.A1_SELECT_FROM_SAFE))
        per_seed.append({"seed": seed, "ungated": ungated, "a1": a1})
        print(f"seed {seed:>2}  episodes={len(eps):<4}  "
              f"ungated coll={ungated['collision_rate']:.3f}  "
              f"A1 coll={a1['collision_rate']:.3f}  "
              f"A1 succ={a1['success_rate']:.3f}")

    summary = summarize_seeds(per_seed)
    return {
        "experiment": "a1_runtime_stability",
        "section": "3.11",
        "policy": "random",
        "dataset": E.run.DEFAULT_DATASET_PATH,
        "seeds": seeds,
        "episodes_requested": episodes,
        "episodes": len(eps),
        "note": "Same frozen-episode prefix on every seed. A1 accepts a "
                "proposal only when the L1 monitor marks it safe, otherwise "
                "the classical planner executes. Ungated executes the random "
                "proposal verbatim. Frozen baselines are not written.",
        "a1_collision_always_zero": summary["a1_collision_always_zero"],
        "summary": {
            "a1": summary["a1"],
            "ungated": summary["ungated"],
        },
        "per_seed": per_seed,
    }


def write_json(payload: dict[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_figure(payload: dict[str, Any], path: str) -> None:
    """Collision rate against seed. A1 should sit on the zero line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["per_seed"]
    seeds = [row["seed"] for row in rows]
    ungated = [row["ungated"]["collision_rate"] for row in rows]
    a1 = [row["a1"]["collision_rate"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(seeds, ungated, "o-", color="#e53935", lw=1.8, ms=6,
            label="ungated (random)")
    ax.plot(seeds, a1, "s-", color="#2e7d32", lw=1.8, ms=6,
            label="A1 (select-from-safe)")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xlabel("Random-policy seed")
    ax.set_ylabel("Collision rate")
    ax.set_ylim(-0.05, 1.08)
    if len(seeds) <= 12:
        ax.set_xticks(seeds)
    ax.set_title("A1 runtime stability: collision rate across seeds (§3.11)",
                 fontsize=11)
    ax.legend(framealpha=0.96)
    fig.tight_layout()

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.seeds < 1 or args.episodes < 1:
        raise SystemExit("error: --seeds and --episodes must be >= 1")
    payload = run_sweep(args.seeds, args.episodes)
    write_json(payload, args.json)
    write_figure(payload, args.figure)

    a1 = payload["summary"]["a1"]["collision_rate"]
    ungated = payload["summary"]["ungated"]["collision_rate"]
    claim = "holds" if payload["a1_collision_always_zero"] else "FAILS"
    print(f"\nA1 collision claim {claim}: "
          f"min={a1['min']:.3f} max={a1['max']:.3f} "
          f"on {payload['seeds']} seeds × {payload['episodes']} episodes")
    print(f"ungated collision mean={ungated['mean']:.3f} "
          f"std={ungated['std']:.3f}")
    print(f"wrote {args.json}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
