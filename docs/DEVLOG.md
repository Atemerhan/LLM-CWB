# Development Log

## 2026-06-08 — U1 self-consistency experiment (full run) — result: AGAINST

**Goal.** Test whether a *derived* uncertainty signal (self-consistency over k
resamples) can serve as the trust signal (U1) the authority arbiter needs to
grant the LLM authority above A1.

**What ran.** `u1_selfconsistency.py` via the manual `U1 Full Run` workflow
(`workflow_dispatch`), commit `ed49d6e`, branch
`claude/confidently-wrong-detection-day1-He1vl`.
- Config (pre-registered, unchanged): n=300 seeded-stratified (100/100/100),
  k=10, temperature 0.7, model `deepseek-chat`, prompt v0, 2000-resample
  bootstrap, α=0.05.
- Run 27167031752: **success**, ~71 min, **3000 API calls, 0 failures/retries**.
- Artifact: `u1-results` (`results/deepseek_u1.json`, 7.6 KB, artifact 7493795160).

**Result.** Decision rule → **AGAINST** (`signal beats safety baseline = NO`,
delta_safety_CI95.hi = +0.030). Key numbers:
- AURC_safety 0.260 (baseline 0.270), Δ CI95 [−0.048, +0.030], p1=0.303.
- AUROC(unsafe) 0.525, CI95 [0.449, 0.598].
- High-agreement(≥0.9) unsafe-rate 0.282 > baseline; **23.8% unsafe at full agreement**.
- Hard slice worse than chance (AURC 0.417 > base 0.390).

**Decision.** Self-consistency rejected as U1; authority ceiling held at **A1**;
SPEC_v2 §9 updated; thesis Ch4/Ch5 + experiment report written.

**Process notes.**
- No benchmark code/dataset/prompt/scoring changed; only the standalone U1 script
  + its workflows + docs.
- Unit tests `test_u1_selfconsistency.py` (7/7 pass) cover the new statistics.
- Branch divergence flagged: U1 lives on the default branch; GLM + original docs
  live on unmerged PR #4 — consolidation pending.

## Prior milestones (abridged)
- 2026-06-05 — DeepSeek v0 full benchmark (1200): acc 0.575, ECE 0.425, CW 42.5%.
- 2026-06-06 — GLM adapter + smoke (PR #4); full GLM run deferred.
- 2026-06-06/08 — Froze SPEC_v2 / ARCHITECTURE; re-scoped thesis to
  trust-/authority-bounded LLM decision-making with runtime assurance.
- 2026-06-08 — U1 pilot (30×10): signal non-degenerate → full run authorized.
