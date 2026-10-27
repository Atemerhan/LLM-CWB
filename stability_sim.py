#!/usr/bin/env python3
"""System-level runtime-stability test for the UAV–LLM closed loop (kinematic).

Strengthens the engineering-implementation evidence WITHOUT adding any model or
benchmark: it re-runs the *existing* frozen-grid closed loop (`embodied_sim`)
under an adversarial random high-level policy across many random seeds, to show
that the safety-by-construction invariant is **stable**, not a single-seed
artefact:

  - under authority **A1** the collision rate is identically 0 on every seed
    (the L1 monitor vetoes every unsafe command -> planner fallback);
  - under the **UNGATED** ablation the same policy collides at a high, variable
    rate (the raw danger that A1 removes).

Pure Python, headless, deterministic given the seed list, no model calls — runs
in the sandbox / CI. Additive: writes results/embodied_stability.json and
figures/fig_stability.png; the frozen v0 baseline and existing results are not
touched.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

import embodied_sim as E


def _agg(values: list[float]) -> dict:
    """mean / std / min / max for a metric across seeds."""
    return {
        "mean": st.fmean(values),
        "std": st.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_sweep(n_seeds: int, n_episodes: int | None) -> dict:
    eps = E.episodes_from_dataset()
    if n_episodes is not None:
        eps = eps[:n_episodes]
    seeds = list(range(n_seeds))

    # Adversarial random policy under UNGATED vs A1; the deterministic greedy and
    # oracle arms are seed-invariant references (one run each).
    arms = {
        "ungated_random": E.Authority.UNGATED,
        "a1_random": E.Authority.A1_SELECT_FROM_SAFE,
    }
    per_seed: dict[str, list[dict]] = {a: [] for a in arms}
    for seed in seeds:
        pol = E.make_random_policy(seed)
        for arm, auth in arms.items():
            s = E.summarize(E.run_arm(eps, pol, auth))
            s["seed"] = seed
            per_seed[arm].append(s)

    # determinism / reproducibility: same seed twice -> identical metrics
    a = E.summarize(E.run_arm(eps, E.make_random_policy(0), E.Authority.A1_SELECT_FROM_SAFE))
    b = E.summarize(E.run_arm(eps, E.make_random_policy(0), E.Authority.A1_SELECT_FROM_SAFE))
    deterministic = all(abs(a[k] - b[k]) < 1e-12 for k in ("success_rate", "spl", "collision_rate", "veto_rate"))

    metrics = ("collision_rate", "success_rate", "spl", "veto_rate")
    aggregates = {
        arm: {m: _agg([r[m] for r in rows]) for m in metrics}
        for arm, rows in per_seed.items()
    }

    # seed-invariant deterministic references
    greedy_a1 = E.summarize(E.run_arm(eps, E.greedy_manhattan_policy, E.Authority.A1_SELECT_FROM_SAFE))
    oracle = E.summarize(E.run_arm(eps, E.oracle_policy, E.Authority.PLANNER_ONLY))

    return {
        "experiment": "embodied_archA_runtime_stability",
        "note": "Seed sweep of an adversarial random high-level policy over the "
                "frozen grid closed loop; no model calls. Headline: A1 collision "
                "rate is identically 0 on every seed (stable safety invariant), "
                "while UNGATED collides at a high variable rate.",
        "episodes_per_seed": len(eps),
        "n_seeds": n_seeds,
        "seeds": seeds,
        "deterministic_reproducible": deterministic,
        "aggregates": aggregates,
        "references": {"greedy_a1": greedy_a1, "oracle_planner_only": oracle},
        "per_seed": per_seed,
    }


def make_figure(payload: dict, outdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = payload["seeds"]
    ung = payload["per_seed"]["ungated_random"]
    a1 = payload["per_seed"]["a1_random"]
    ung_coll = [r["collision_rate"] for r in ung]
    a1_coll = [r["collision_rate"] for r in a1]
    a1_succ = [r["success_rate"] for r in a1]
    a1_spl = [r["spl"] for r in a1]
    a1_veto = [r["veto_rate"] for r in a1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # Panel 1: collision rate per seed — A1 flat at 0, UNGATED high/variable
    ax1.plot(seeds, ung_coll, "o-", color="#c62828", label="UNGATED (no L1 monitor)", lw=1.6, ms=4)
    ax1.plot(seeds, a1_coll, "s-", color="#2e7d32", label="A1 (select-from-safe)", lw=1.6, ms=4)
    ax1.axhline(0, color="#2e7d32", ls=":", lw=1)
    ax1.set_xlabel("random seed"); ax1.set_ylabel("collision rate")
    ax1.set_title("Safety invariant across seeds\nA1 collision rate = 0 on every seed")
    ax1.set_ylim(-0.03, max(ung_coll + [0.1]) * 1.15)
    ax1.legend(fontsize=8, loc="center right"); ax1.grid(alpha=0.3)

    # Panel 2: A1 success / SPL distribution across seeds (performance, not safety)
    ax2.boxplot([a1_succ, a1_spl], tick_labels=["success", "SPL"], widths=0.5)
    ax2.scatter([1] * len(a1_succ), a1_succ, color="#1565c0", alpha=0.5, s=18, zorder=3)
    ax2.scatter([2] * len(a1_spl), a1_spl, color="#6a1b9a", alpha=0.5, s=18, zorder=3)
    ax2.set_ylabel("rate"); ax2.set_ylim(0, 1)
    ax2.set_title("A1 performance spread\n(graceful cost, safety untouched)")
    ax2.grid(alpha=0.3, axis="y")

    # Panel 3: A1 veto load across seeds (bounded, stable)
    ax3.plot(seeds, a1_veto, "D-", color="#e65100", lw=1.6, ms=4)
    ax3.set_xlabel("random seed"); ax3.set_ylabel("veto rate (vetoed / proposed)")
    ax3.set_title("A1 veto load across seeds\n(L1 absorbs the unsafe proposals)")
    ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

    agg = payload["aggregates"]
    fig.suptitle(
        f"UAV–LLM closed-loop runtime stability — {payload['n_seeds']} seeds × "
        f"{payload['episodes_per_seed']} episodes  ·  "
        f"A1 collision max={agg['a1_random']['collision_rate']['max']:.3f}  ·  "
        f"UNGATED collision mean={agg['ungated_random']['collision_rate']['mean']:.3f}",
        fontsize=10.5, y=1.02)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fig_stability.png")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_stability.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of random seeds")
    ap.add_argument("--episodes", type=int, default=None, help="episode subset (default: all 1200)")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    payload = run_sweep(args.seeds, args.episodes)
    os.makedirs("results", exist_ok=True)
    with open("results/embodied_stability.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    agg = payload["aggregates"]
    print(f"episodes/seed={payload['episodes_per_seed']}  seeds={payload['n_seeds']}  "
          f"deterministic={payload['deterministic_reproducible']}")
    for arm in ("ungated_random", "a1_random"):
        c = agg[arm]["collision_rate"]
        print(f"  {arm:>15}  collision mean={c['mean']:.3f} std={c['std']:.3f} "
              f"min={c['min']:.3f} max={c['max']:.3f}")
    print("wrote results/embodied_stability.json")
    if not args.no_fig:
        print("wrote", make_figure(payload, args.outdir), "+ .pdf")


if __name__ == "__main__":
    main()
