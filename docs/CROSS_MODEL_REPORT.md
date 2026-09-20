# Cross-Model Report (SP2 — n=3 replication)

Three off-the-shelf LLMs through the **identical frozen instrument** (prompt v0,
1200-sample dataset, BFS oracle, scoring, calibration, U1 protocol, pre-registered
decision rule). Only the queried model differs. All numbers trace to committed
`results/*.json`.

| Model | Model id | Provider |
| --- | --- | --- |
| DeepSeek | `deepseek-chat` | DeepSeek |
| GLM | `glm-4-flash` | Zhipu |
| Kimi | `moonshot-v1-8k` | Moonshot |

## 1. Accuracy
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| Overall | 0.575 | 0.315 | 0.335 |
| easy / medium / hard | 0.635 / 0.650 / 0.440 | 0.275 / 0.365 / 0.305 | 0.292 / 0.380 / 0.333 |
DeepSeek is the strongest; GLM and Kimi cluster ~0.32. GLM/Kimi are **non-monotone**
in difficulty (medium > easy), indicating systematic grid-misreading, not hardness.

## 2. Calibration & ECE
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| Mean confidence | 100.0 | 100.0 | 100.0 |
| ECE | 0.425 | 0.685 | 0.665 |
| Distinct confidence buckets | 1 ([90,100]) | 1 | 1 |
All three saturate confidence at 100 → the reliability diagram collapses to one
bucket and **ECE = error rate** for every model. Verbalized confidence is
degenerate across all three providers.

## 3. Confidence distribution
Identical across models: a single spike at 100 (mean 100.0, no spread). The
distribution itself is the calibration-failure evidence (Figure 6).

## 4. Confidently-wrong rate
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| CW count / rate | 510 / **42.5%** | 822 / **68.5%** | 798 / **66.5%** |
| Malformed rate | 0.000 | 0.000 | 0.000 |
ED replicates **n=3**; severity tracks (in)competence — the weaker models are
more confidently wrong at identical (maximal) expressed confidence.

## 5. Unsafe rate & safety gate (U1)
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| Baseline unsafe-rate | 0.270 | 0.350 | 0.323 |
| Unsafe-rate @ agreement ≥0.9 | 0.282 | 0.343 | 0.273 |
| High-agreement coverage | 0.343 | 0.330 | **0.073** |
| Unsafe-rate @ agreement =1.0 (n) | 0.238 (42) | 0.355 (31) | 0.400 (5) |
No model's high-agreement decisions are reliably safer; Kimi *appears* lower
(0.273) but its CI [0.132,0.482] includes baseline and coverage is only 7%.

## 6. AUROC(unsafe) — does agreement rank safety?
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| AUROC | 0.525 | 0.505 | **0.589** |
| 95% CI | [0.449,0.598] | [0.437,0.573] | [0.523,0.656] |
DeepSeek/GLM ≈ chance. **Kimi is weakly above chance** (lower CI 0.523 > 0.5) —
the one partial signal — but see §8.

## 7. Agreement statistics & tie diagnostic
| | DeepSeek | GLM | Kimi |
| --- | --- | --- | --- |
| Agreement mean (std) | 0.738 (0.196) | 0.773 (0.148) | **0.589 (0.180)** |
| Agreement on committed | 0.738 | 0.773 | 0.589 |
| Agreement on correct-set | 0.545 | 0.316 | 0.348 |
| Tie gap | −0.193 | −0.457 | −0.240 |
All gaps negative ⇒ every model is more self-consistent about its *committed*
(often wrong) move than about *correct* moves → **systematic errors**. DeepSeek
/GLM are highly self-consistent; Kimi is markedly **less** self-consistent.

## 8. Decision-rule verdicts (pre-registered, frozen)
| FOR criterion | DeepSeek | GLM | Kimi |
| --- | :--: | :--: | :--: |
| Δ(AURC_safety) CI95 upper < 0 | ✗ | ✗ | ✓ |
| AUROC lower > 0.5 | ✗ | ✗ | ✓ |
| high-agreement unsafe < baseline → B_target | ✗ | ✗ | ✗ |
| coverage ≥ 0.30 | ✓ | ✓ | ✗ |
| hard-slice Δ CI upper < 0 | ✗ | ✗ | ✗ |
| **Verdict** | **AGAINST** | **AGAINST** | **AGAINST** |

## 9. Failure modes (why each fails)
- **DeepSeek & GLM:** confidently self-consistent about wrong moves; agreement at
  chance for safety (AUROC ≈ 0.5); 23.8% / 35.5% unsafe at unanimous agreement.
- **Kimi:** hesitant (low self-agreement 0.589); a faint global signal (AUROC
  0.589, AURC Δ p=0.021) that is **operationally unusable** — only 7% of
  decisions reach high agreement and the signal vanishes on the hard slice.
- **Common:** verbalized confidence degenerate (ECE = error rate); self-consistency
  fails the safety gate; errors systematic (negative tie gap).

## 10. Conclusion
**3/3 models AGAINST** under one pre-registered rule, via *different* mechanisms —
a robustness result: the conclusion holds whether the model is over-consistent
(DeepSeek/GLM) or under-consistent with a faint signal (Kimi). Across three
independent providers, **neither reported nor sampled self-signals certify
grounded decisions**, so the maximum trustworthy authority is **A1**
(select-from-safe) behind a verified planner and runtime monitor.

---
*Artifacts:* `results/{deepseek,glm,kimi}.json` + `*_u1.json` (see
`BASELINE_FREEZE_REPORT.md` for commit hashes). Frozen prompt v0, dataset v1,
scoring, U1 rule.
