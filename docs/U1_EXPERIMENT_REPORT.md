# Experiment Report — U1: Self-Consistency as an Uncertainty Signal for Safe LLM Decision Authority

## Objective
Determine whether a *derived* uncertainty signal — self-consistency (agreement
across k stochastic resamples) — can serve as the trust signal that a
runtime-assured UAV-navigation architecture needs to grant an LLM decision
authority above level A1 (i.e., to operate a confidence-gated deferral policy that
reduces collision risk).

## Hypothesis
- **H1:** Higher self-consistency agreement is associated with fewer unsafe
  (collision) decisions strongly enough that a confidence-gated policy makes the
  acted-on decisions meaningfully safer than acting on all decisions.
- **H0 (null):** Agreement carries no usable safety information — the most
  confident decisions are no safer than average (constant-confidence baseline).

## Methodology
- **Task:** single-step next-move prediction on 8×8 grid mazes; a move is *unsafe*
  iff its destination is off-grid, into a wall, or onto an unreachable cell
  (collision), determined by a breadth-first-search oracle (frozen `run.py`).
- **Model:** DeepSeek `deepseek-chat` (matches the committed temp-0 baseline),
  prompt **v0** (frozen), via an OpenAI-compatible API with transient-failure
  retry in the experiment wrapper only.
- **Procedure:** for each item, build the identical v0 prompt and sample **k=10**
  responses at **temperature 0.7**; parse each with the frozen parser; define
  *agreement* = fraction of the k valid votes equal to the committed temp-0
  decision; that decision's *unsafe*/*incorrect* labels come from the oracle.
- **Inference:** tie-corrected AURC over the unsafe (and incorrect) indicator with
  a **2000-resample nonparametric bootstrap** 95% CI and a paired one-sided test
  vs the constant-confidence baseline (= mean loss); AUROC(unsafe) (midrank
  Mann-Whitney) with bootstrap CI; Wilson 95% CIs on per-bin and gate rates; **α =
  0.05**. Spearman/Kendall reported **exploratory only** (excluded from the rule).
- **Pre-registered decision rule (FOR requires all):** Δ(AURC_safety) CI95 upper
  < 0; AUROC(unsafe) CI95 lower > 0.5; high-agreement (≥0.9) unsafe-rate < baseline
  (target B_target = 0.5×baseline); coverage ≥ C_min = 0.30; **hard-slice
  REQUIRED** (its Δ CI upper < 0). Any failure ⇒ not FOR; the AGAINST pattern is a
  null effect and/or operational failure.
- **Execution:** manual `workflow_dispatch` (`U1 Full Run`), commit `ed49d6e`,
  run 27167031752, **3000 API calls**, ~71 min, 0 failures/retries. The pilot
  (30×10) had previously cleared the degeneracy gate.

## Dataset
Committed evaluation set `data/eval_dataset.json` (1200 samples, 8×8 grids, wall
density 0.25, seed 42, balanced 400/400/400 across easy/medium/hard). U1 uses a
**seeded-random stratified subset of n=300** (100 per difficulty, seed 20240608);
the committed DeepSeek temp-0 decisions (`results/deepseek.json`) provide each
item's reference move and labels. Subset is deterministic and recorded in the
artifact (`meta.subset_ids`).

## Metrics
Primary: **AURC_safety** (area under the unsafe-rate vs coverage curve; lower
better) with bootstrap CI and Δ-vs-baseline test; **AUROC(unsafe)**; **high-
agreement unsafe-mass** (acted-on collision rate at agreement ≥0.9, with coverage).
Secondary: AURC_correctness; per-difficulty AURC_safety; Wilson-CI agreement bins;
degeneracy diagnostics; tie-aware diagnostic (oracle, non-deployable). Exploratory:
Spearman/Kendall.

## Results (n=300)
- **Signal non-degenerate:** agreement mean 0.738, std 0.196, 86% with ≥1
  disagreement.
- **No discrimination:** AURC_safety 0.260 vs baseline 0.270, Δ=−0.010, 95% CI
  **[−0.048, +0.030]**, p₁=0.303; AUROC(unsafe) 0.525, 95% CI **[0.449, 0.598]**
  (both include the null).
- **Operational failure:** unsafe-rate at agreement ≥0.9 = **0.282** (CI [0.204,
  0.375]) ≥ baseline 0.270 ≫ B_target 0.135; **23.8% unsafe at agreement = 1.0**;
  coverage 0.343.
- **Hard slice worse than chance:** AURC_safety 0.417 > baseline 0.390 (Δ CI up
  +0.103) — required criterion fails.
- **Tie diagnostic:** agreement(committed) 0.738 > agreement(correct-set) 0.545
  (gap −0.193), inflation 16.3% — confidence is genuine commitment to wrong moves,
  not multi-optimal ambiguity.
- **Verdict:** every FOR criterion except coverage fails ⇒ **AGAINST**.

## Discussion
Self-consistency captures output *stability*, not *competence*. The LLM's errors
are systematic — it is more self-consistent than correct and repeats the same
misreadings — so agreement cannot separate safe from unsafe and, at high
confidence, is no safer than chance. Because the k samples share the model's blind
spots, their agreement overstates reliability by construction. The result extends
the Embodied-Divergence finding from self-*reported* confidence to a *derived*
signal: the divergence is **not self-detectable**, so an off-the-shelf LLM cannot
self-gate safety. The practical consequence is a measured **authority ceiling of
A1**: the LLM is bounded to selecting among planner-supplied safe options, with all
hard guarantees held by the verified planner and the L1 monitor.

## Threats to validity
- **Construct/scope:** rejects self-consistency at k=10, T=0.7 only; logprob,
  ensemble, and learned-verifier signals are untested and may differ.
- **External validity:** single model, single task, single prompt/temperature;
  fully-observed geometric grids (the regime where classical planners dominate and
  an LLM is least needed).
- **Statistical:** k=10 yields a coarse, noisy per-item confidence (11 levels);
  n=300 + bootstrap mitigate but a small true effect could be missed. The decisive
  *operational* failure (high-agreement unsafe-rate ≥ baseline) does not depend on
  statistical power.
- **Measurement:** the confidence (T=0.7 agreement) labels a T=0 decision —
  ordinal, not a calibrated probability. Single-step task; no vehicle dynamics,
  disturbances, partial observability, or trajectory-level safety. Per-step unsafe
  is a proxy for mission collision risk.
- **Monitor assumption:** safety-by-construction assumes a correct L1 monitor
  (not formally verified here).

## Conclusion
Under a pre-registered decision rule, **self-consistency is rejected as a usable
safety/trust signal (AGAINST)**. Embodied Divergence is not self-detectable even by
sampling; the LLM's safe decision authority is bounded at **A1**; and the UAV
architecture must keep the LLM advisory behind a verified planner and runtime
safety monitor. The contribution is the reproducible, measurement-driven method
that produced this authority bound — a defensible negative result for an
engineering thesis on trustworthy LLM-in-the-loop autonomy.

---
*Run:* 27167031752 · *commit:* `ed49d6e` · *artifact:* `u1-results`
(`results/deepseek_u1.json`) · *config:* deepseek-chat, T=0.7, k=10, n=300, seed
20240608, 2000-bootstrap, α=0.05.
