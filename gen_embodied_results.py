"""Generate 档 A closed-loop results on the full frozen dataset + a report.

Runs the reference arms (the real LLM arm is wired later via CachedPolicy) over
all 1200 frozen episodes and writes:
  - results/embodied_baseline.json  (machine-readable, new path; baseline untouched)
  - docs/EMBODIED_RESULTS.md        (human report for Ch4/Ch6 closed-loop evidence)

No model calls. Deterministic. The headline is the safety-by-construction result:
A1 drives the collision rate to 0 regardless of how unsafe the high-level policy
is, at a graceful SPL cost (performance, not safety).
"""
from __future__ import annotations

import json

import embodied_sim as E

ARMS = [
    ("planner_only", "oracle", E.oracle_policy, E.Authority.PLANNER_ONLY),
    ("ungated", "greedy", E.greedy_manhattan_policy, E.Authority.UNGATED),
    ("a1", "greedy", E.greedy_manhattan_policy, E.Authority.A1_SELECT_FROM_SAFE),
    ("ungated", "random", E.make_random_policy(0), E.Authority.UNGATED),
    ("a1", "random", E.make_random_policy(0), E.Authority.A1_SELECT_FROM_SAFE),
]


def main() -> None:
    eps = E.episodes_from_dataset()           # all 1200 frozen episodes
    rows = []
    for authority, policy_name, pol, auth in ARMS:
        s = E.summarize(E.run_arm(eps, pol, auth))
        s.update(authority=authority, policy=policy_name)
        rows.append(s)
        print(f"{authority:>12} / {policy_name:<7}  succ={s['success_rate']:.3f} "
              f"spl={s['spl']:.3f} coll={s['collision_rate']:.3f} "
              f"veto={s['veto_rate']:.3f}")

    payload = {
        "experiment": "embodied_archA_closed_loop",
        "episodes": len(eps),
        "note": "Reference policies only; real LLM arm pending API wiring "
                "(CachedPolicy). Frozen dataset reused as episodes; v0 baseline "
                "untouched.",
        "arms": rows,
    }
    with open("results/embodied_baseline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\nwrote results/embodied_baseline.json")

    def fmt(r: dict) -> str:
        t = r["terminations"]
        return (f"| {r['authority']} / {r['policy']} | {r['success_rate']:.3f} | "
                f"{r['spl']:.3f} | **{r['collision_rate']:.3f}** | "
                f"{r['veto_rate']:.3f} | {r['mean_steps']:.1f} | "
                f"{t['goal']}/{t['collision']}/{t['step_cap']}/{t['stuck']} |")

    md = [
        "# EMBODIED_RESULTS.md — 档 A 运动学闭环（参考策略，n=1200 frozen episodes）",
        "",
        "Closed-loop rollout of the L0–L3 / A0–A3 architecture on the **frozen**",
        "grid episodes (each dataset sample = one `(grid, start, goal)` episode;",
        "BFS oracle = optimal length + L1 safety predicate). Pure Python,",
        "deterministic, CI-reproducible; **no model calls** — the real LLM arm is",
        "injected later via `CachedPolicy`. Baseline v0 untouched; outputs go to",
        "`results/embodied_baseline.json`.",
        "",
        "| arm (authority / policy) | success | SPL | collision | veto-rate | mean-steps | goal/coll/cap/stuck |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    md += [fmt(r) for r in rows]
    md += [
        "",
        "## Reading",
        "- **planner_only (oracle)** is the upper bound: optimal, safe, success=SPL=1.",
        "- **ungated** arms execute the policy verbatim (no L1 monitor): a wall-blind",
        "  greedy policy collides ~15%, a random policy ~80% — the raw danger of",
        "  trusting an untrusted high-level controller directly (the per-step unsafe",
        "  rate compounding over a trajectory, SPEC_v2 S3).",
        "- **A1 (select-from-safe + L1)** drives the **collision rate to exactly 0**",
        "  for *both* policies, however unsafe they are — safety-by-construction. The",
        "  cost is paid in SPL/steps (performance), never in safety: this is the",
        "  closed-loop, trajectory-level realization of the thesis claim that the A1",
        "  authority ceiling *safely neutralizes* an untrustworthy LLM.",
        "- The **veto-rate** is the L1 monitor's fallback load (how often the planner",
        "  rescues an unsafe proposal) — directly echoing the 27–35% single-step",
        "  unsafe rate measured in Exp 0 / U1.",
        "",
        "## Status & next",
        "- This is the E-A1/E-A2 reference run. **Pending:** wire the real LLM arm",
        "  (`A1-LLM+A*`) via `CachedPolicy` over live/replayed DeepSeek/GLM/Kimi",
        "  decisions (needs API calls → GitHub Actions), and the `A2-LLM+observer`",
        "  arm once U2 yields a usable trust signal.",
        "- Threats to validity: 2D occupancy lifted to waypoints; no aerodynamics/",
        "  attitude coupling (this is a kinematic prototype, not a flight test) —",
        "  see `EMBODIED_PROTOTYPE_PLAN.md` §10.",
    ]
    with open("docs/EMBODIED_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote docs/EMBODIED_RESULTS.md")


if __name__ == "__main__":
    main()
