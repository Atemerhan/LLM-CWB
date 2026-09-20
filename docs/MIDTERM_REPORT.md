# Mid-term Report Update

**Project:** LLM-Based Decision-Making for UAV Navigation under a Safety-Bounded
Hierarchical Architecture with Confidence-Gated Runtime Assurance.

## 1. Progress since proposal
- Built a reproducible grid-navigation benchmark with a BFS oracle (ground-truth
  optimal/safe reference), confidence protocol, and scoring (frozen).
- Characterized **Embodied Divergence** on an off-the-shelf LLM (DeepSeek):
  accuracy 0.575, **ECE 0.425**, **42.5% confidently-wrong**, with self-reported
  confidence saturated at 100 (non-informative).
- Designed and froze the evaluation framework (SPEC_v2: Performance / Reliability
  / Safety / Latency / Trust) and the L0–L3 runtime-assurance architecture with
  an A0–A3 authority ladder and a pre-registered authority-assignment rubric.
- Ran the **U1 experiment** (self-consistency as a derived uncertainty signal)
  to decide whether the LLM can be trusted above authority level A1.

## 2. Key result (U1)
A pre-registered test (n=300, k=10, T=0.7, 2000-bootstrap, α=0.05) returned
**AGAINST**: self-consistency does **not** discriminate unsafe decisions
(AURC_safety Δ CI95 [−0.048, +0.030], AUROC 0.525 [0.449, 0.598]), and the
most-confident decisions are **not** safer (unsafe-rate 0.282 at agreement ≥0.9
vs 0.270 baseline; **23.8% unsafe even at unanimous agreement**). On the hard
slice it is worse than chance.

## 3. Interpretation
- The LLM's errors are **systematic, not stochastic** — it is more self-consistent
  (0.738) than correct (~0.59) and repeats the same wrong/unsafe move across
  resamples. Self-consistency therefore cannot flag them.
- **Authority ceiling is empirically pinned at A1.** Safety remains guaranteed by
  the verified planner + L1 monitor, not by the LLM.
- This is a **publishable negative result** that strengthens the Embodied-Divergence
  thesis: even a *derived* uncertainty signal fails to make the divergence
  self-detectable.

## 4. Revised plan for the second half
1. Document and integrate U1 (this update; thesis Ch4/Ch5; SPEC_v2 §9). — DONE/IN PROGRESS
2. (Optional, future) Alternative U1 signals — token-logprob (T=0), ensemble
   disagreement, learned verifier — which could re-open authority level A2.
3. Closed-loop episode evaluation (SPL) of A1-bounded LLM vs pure A\*.
4. Consolidate branches (merge GLM/docs PR with the U1 work).

## 5. Risk / honesty note
On a fully-observed grid a classical planner already dominates; the contribution
is the **measurement-driven authority-bounding methodology**, not an LLM
performance win. The A1 ceiling is the methodology working as intended.
