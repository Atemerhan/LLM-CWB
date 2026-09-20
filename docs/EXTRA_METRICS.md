# EXTRA_METRICS.md — P2 step-regret + S3 cumulative collision (read-only)

Additive, read-only metrics from the committed `results/<model>.json` via
the frozen BFS oracle. No model calls; v0 baseline untouched.

## P2 — Step-regret distribution (over well-formed, reachable moves)
ρ = d(dest) − (d(pos) − 1).  Grid parity: adjacent cells differ in BFS
distance by an odd amount, so ρ is always **even** — ρ=0 correct (toward),
ρ=2 one step away, ρ≥4 further. (Odd ρ is structurally impossible.)

| Model | n | mean | median | p95 | correct(0) | away(2) | far(≥4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek deepseek-chat | 882 | 0.435 | 0.0 | 2 | 690 | 192 | 0 |
| GLM glm-4-flash | 778 | 1.028 | 2.0 | 2 | 378 | 400 | 0 |
| Kimi moonshot-v1-8k | 801 | 0.996 | 0.0 | 2 | 402 | 399 | 0 |

*Reading:* regret grades the error that binary accuracy hides. A clean
finding: among well-formed *safe* moves there is essentially no ρ≥4 — when
a model is wrong-but-safe it retreats by exactly one step (ρ=2), not
wildly. So the two failure modes are sharp: a safe one-step retreat, or an
outright collision (S1/S5, excluded here) — consistent with systematic,
confident misreading rather than diffuse noise.

## S3 — Cumulative collision probability projection  1 − (1 − U)^L
U = per-step unsafe rate (well-formed). Projects the single-step collision
proxy over a trajectory of length L (independence approximation).

| Model | U (per-step) | L=1 | L=2 | L=3 | L=5 | L=10 | L=20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek deepseek-chat | 0.265 | 0.265 | 0.460 | 0.603 | 0.785 | 0.954 | 0.998 |
| GLM glm-4-flash | 0.352 | 0.352 | 0.580 | 0.727 | 0.885 | 0.987 | 1.000 |
| Kimi moonshot-v1-8k | 0.333 | 0.333 | 0.554 | 0.703 | 0.867 | 0.982 | 1.000 |

*Reading:* even a moderate per-step unsafe rate compounds fast — by a
10-step path every model exceeds ~0.95 cumulative collision probability if
executed ungated. This is exactly the danger the closed-loop §4.7 removes:
under A1 the realized collision rate is 0 regardless of U. (Independence is
an approximation; the measured closed-loop UNGATED rates 0.59/0.94 are the
empirical counterpart.)

## Calibration note — AURC under saturated confidence
Because verbalized confidence is saturated at 100 for all three models
(§4.2), ordering by confidence is uninformative: the selective-accuracy
risk–coverage curve is flat and **AURC_acc(confidence) = overall error
rate** (0.425 / 0.685 / 0.665). I.e. confidence-based selective prediction
yields no gain — the quantitative restatement of 'confidence carries no
usable information'.
