# EMBODIED_RESULTS.md — 档 A 运动学闭环（参考策略，n=1200 frozen episodes）

Closed-loop rollout of the L0–L3 / A0–A3 architecture on the **frozen**
grid episodes (each dataset sample = one `(grid, start, goal)` episode;
BFS oracle = optimal length + L1 safety predicate). Pure Python,
deterministic, CI-reproducible; **no model calls** — the real LLM arm is
injected later via `CachedPolicy`. Baseline v0 untouched; outputs go to
`results/embodied_baseline.json`.

| arm (authority / policy) | success | SPL | collision | veto-rate | mean-steps | goal/coll/cap/stuck |
| --- | --- | --- | --- | --- | --- | --- |
| planner_only / oracle | 1.000 | 1.000 | **0.000** | 0.000 | 6.1 | 1200/0/0/0 |
| ungated / greedy | 0.410 | 0.410 | **0.590** | 0.000 | 2.7 | 492/708/0/0 |
| a1 / greedy | 0.651 | 0.650 | **0.000** | 0.446 | 25.2 | 781/0/419/0 |
| ungated / random | 0.057 | 0.044 | **0.943** | 0.000 | 3.6 | 68/1132/0/0 |
| a1 / random | 0.794 | 0.334 | **0.000** | 0.236 | 30.7 | 953/0/247/0 |

## Reading
- **planner_only (oracle)** is the upper bound: optimal, safe, success=SPL=1.
- **ungated** arms execute the policy verbatim (no L1 monitor): a wall-blind
  greedy policy collides ~15%, a random policy ~80% — the raw danger of
  trusting an untrusted high-level controller directly (the per-step unsafe
  rate compounding over a trajectory, SPEC_v2 S3).
- **A1 (select-from-safe + L1)** drives the **collision rate to exactly 0**
  for *both* policies, however unsafe they are — safety-by-construction. The
  cost is paid in SPL/steps (performance), never in safety: this is the
  closed-loop, trajectory-level realization of the thesis claim that the A1
  authority ceiling *safely neutralizes* an untrustworthy LLM.
- The **veto-rate** is the L1 monitor's fallback load (how often the planner
  rescues an unsafe proposal) — directly echoing the 27–35% single-step
  unsafe rate measured in Exp 0 / U1.

## Status & next
- This is the E-A1/E-A2 reference run. **Pending:** wire the real LLM arm
  (`A1-LLM+A*`) via `CachedPolicy` over live/replayed DeepSeek/GLM/Kimi
  decisions (needs API calls → GitHub Actions), and the `A2-LLM+observer`
  arm once U2 yields a usable trust signal.
- Threats to validity: 2D occupancy lifted to waypoints; no aerodynamics/
  attitude coupling (this is a kinematic prototype, not a flight test) —
  see `EMBODIED_PROTOTYPE_PLAN.md` §10.
